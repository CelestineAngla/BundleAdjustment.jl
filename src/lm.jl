using SparseArrays
using SolverTools
using NLPModels
using AMD
using Metis
using MatrixMarket
include("lma_aux.jl")
include("ldl_aux.jl")
include("qr_aux.jl")

"""
Implementation of Levenberg Marquardt algorithm for NLSModels
Solves min 1/2 ||r(x)||² where r is a vector of residuals
"""
function Levenberg_Marquardt(model :: AbstractNLSModel,
                             facto :: Symbol,
                             perm :: Symbol,
                             normalize :: Symbol,
                             linesearch :: Bool;
                             x :: AbstractVector=copy(model.meta.x0), facto_type :: DataType=eltype(x),
                             restol=eltype(x)(eps(eltype(x))^(1/3)),
                             satol=sqrt(eps(eltype(x))), srtol=sqrt(eps(eltype(x))),
                             oatol=sqrt(eps(eltype(x))), ortol=eltype(x)(eps(eltype(x))^(1/3)),
                             atol=sqrt(eps(eltype(x))), rtol=eltype(x)(eps(eltype(x))^(1/3)),
                             νd=eltype(x)(3), νm=eltype(x)(3), λ=eltype(x)(30), δd=eltype(x)(2),
                             ite_max :: Int=200, max_time :: Int=3600)

  @info model
  @info "Parameters of the solver:\n" facto perm normalize linesearch νd νm λ ite_max facto_type
  @info "Tolerances:\n" restol satol srtol oatol ortol atol rtol

  start_time = time()
  elapsed_time = 0.0
  iter = 0
  x_suiv = similar(x)
  T = eltype(x)

  # Initialize residuals
  r = residual(model, x)
  norm_r = norm(r)
  obj = norm_r^2 / 2
  r_suiv = copy(r)

  # Initialize b = [-r; 0]
  b = Vector{T}(undef, model.nls_meta.nequ + model.meta.nvar)
  b[1 : model.nls_meta.nequ] .= -r
  b[model.nls_meta.nequ + 1 : end] .= 0
  xr = similar(b, facto_type)

  # Initialize J in the format J[rows[k], cols[k]] = vals[k]
  rows = Vector{Int}(undef, model.nls_meta.nnzj)
  cols = Vector{Int}(undef, model.nls_meta.nnzj)
  jac_structure_residual!(model, rows, cols)
  vals = jac_coord_residual(model, x)

  # Compute Jᵀr and λ
  Jtr = mul_sparse(cols, rows, vals, r, model.nls_meta.nnzj, model.meta.nvar)
  norm_Jtr = norm(Jtr)
  λ = T(max(λ, 1e10 / norm_Jtr))

  if facto == :QR
    # Initialize A = [J; √λI] as a sparse matrix
    rows_A = vcat(rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    cols_A = vcat(cols, collect(1 : model.meta.nvar))
    vals_A = vcat(vals, fill(sqrt(λ), model.meta.nvar))
    A = sparse(rows_A, cols_A, vals_A)

  elseif facto == :LDL
    # Initialize A = [[I J]; [Jᵀ - λI]] as sparse upper-triangular matrix
    cols_J = copy(cols)
    cols .+= model.nls_meta.nequ
    rows_A = vcat(collect(1 : model.nls_meta.nequ), rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    cols_A = vcat(collect(1 : model.nls_meta.nequ), cols, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    if normalize == :A
      # In this case we solve [[√λI J]; [Jᵀ -√λI]] [δr; √λδ] = [-√λr; 0]
      b[1 : model.nls_meta.nequ] *= sqrt(λ)
      vals_A = vcat(fill(sqrt(λ), model.nls_meta.nequ), vals, fill(-sqrt(λ), model.meta.nvar))
    else
      vals_A = vcat(fill(T(1.0), model.nls_meta.nequ), vals, fill(-λ, model.meta.nvar))
    end
    A = sparse(rows_A, cols_A, vals_A)

    # Compute permutation
    if perm == :AMD
      P = amd(A)
    elseif perm == :Metis
      P , _ = Metis.permutation(A' + A)
    end
	# P = collect(1 : model.meta.nvar + model.nls_meta.nequ)

	# Perform symbolic analysis of A in facto_type and create matrix in facto_type for the conversion
    if facto_type == Float16
      D = similar(b)
      A_norm = sparse(rows_A, cols_A, Float16(1.0))
      A_bis = copy(A)
    elseif facto_type == Float32
	  A_F32 = sparse(rows_A, cols_A, Float32(1.0))
    end
    ldl_symbolic = ldl_analyse(A, P, upper=true, n=model.meta.nvar + model.nls_meta.nequ, type=facto_type)
  end

  if normalize != :None && facto_type != Float16
    col_norms = Vector{T}(undef, model.meta.nvar)
  end

  local norm_δ, δr2
  ϵ_first_order = atol + rtol * norm_Jtr
  old_obj = obj

  # Stopping criteria
  small_step = false
  first_order = norm_Jtr < ϵ_first_order
  small_residual = norm_r < restol
  small_obj_change =  false
  tired = iter > ite_max || elapsed_time > max_time
  fail = false
  fail2 = false
  status = :unknown

  @info log_header([:iter, :f, :Δf, :dFeas, :λ, :δ, :ρ, :status], [Int, T, T, T, T, T, T, String],
                   hdr_override = Dict(:f => "f(x)", :dFeas => "‖Jᵀr‖", :δ => "‖δ‖"))

  while !(small_step || first_order || small_residual || small_obj_change || tired || fail || fail2)
	if facto_type == Float64
	  t0_iter = time()
    end
    iter += 1

    if facto == :QR
      # We normalize the whole matrix A or only the J part
      if normalize == :A
        normalize_qr_a!(A, col_norms, model.meta.nvar)
      elseif normalize == :J
        normalize_qr_j!(A, col_norms, model.meta.nvar)
      end

      # Solve min ‖[J √λI] δ + [r 0]‖² with QR factorization
      if perm == :AMD
        QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_AMD)
      elseif perm == :Metis
        QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_METIS)
      end
      δ = solve_qr!(model.nls_meta.nequ + model.meta.nvar, model.meta.nvar, xr, b, QR.Q, QR.R, QR.prow, QR.pcol)

      # We denormalize δ and A or J
      if normalize != :None
        denormalize_vect!(δ, col_norms, model.meta.nvar)
        denormalize_qr!(A, col_norms, model.meta.nvar)
      end

      # δr2 = ‖Jδ + r‖²/2
      δr2 = norm(A[1 : model.nls_meta.nequ, :] * δ + r)^2 / 2

    elseif facto == :LDL
      # We normalize J in the matrix A
      if normalize != :None && facto_type != Float16
        normalize_ldl!(A, col_norms, model.meta.nvar, model.nls_meta.nequ)
      end

	  if facto_type == Float64
  	    t0_facto = time()
      end

      # Factorize A and if needed normalize and convert A in facto_type
      if facto_type == Float16
        A_bis.nzval .= A.nzval
		normalize_F16!(A_bis, D, A_norm, model.nls_meta.nequ + model.meta.nvar, Float16)
        LDLT = ldl_factorize(A_norm, ldl_symbolic, true)
  	    normalize_vect!(xr, b, D, model.nls_meta.nequ, model.meta.nvar)
	  elseif facto_type == Float32
		@views A_F32.nzval .= Float32.(A.nzval)
		LDLT = ldl_factorize(A_F32, ldl_symbolic, true)
		xr .= b
      else
        LDLT = ldl_factorize(A, ldl_symbolic, true)
		xr .= b
      end

	  # Solve [[I J]; [Jᵀ - λI]] X = [-r; 0] with LDL factorization
      ldl_solve!(model.nls_meta.nequ + model.meta.nvar, xr, LDLT.L.colptr, LDLT.L.rowval, LDLT.L.nzval, LDLT.D, P)

	  if facto_type == Float64
  	    t_facto = time() - t0_facto
		@info "\nTime facto: " t_facto
      end

	  # Convert δ if needed
	  # if T != facto_type
	  #   xr = convert(Array{T, 1}, xr)
	  # end

	  # # Perform iterative refinement
	  # if iterative_refinement
		# step_accepted = true
		# nb_ir = 0
		# norm_res = norm(b - A * xr)
		# tol = eps(T)^(1/25) * norm(b)
		# while norm_res > tol && nb_ir < 5
		#   nb_ir += 1
	  #     xr2_F32 = facto_type.(b - A * xr)
		#   ldl_solve!(model.nls_meta.nequ + model.meta.nvar, xr2_F32, LDLT.L.colptr, LDLT.L.rowval, LDLT.L.nzval, LDLT.D, P)
		#   xr2 = convert(Array{T, 1}, xr2_F32)
		#   xr += xr2
		#   norm_res = norm(b - A * xr)
	  #   end
		# if norm_res > tol
		#   step_accepted = false
	  #   end
	  # end

	  # if iterative_refinement
		# nb_ir = 0
		# while norm(b - A * xr) > 1600 && nb_ir < 5
		#   nb_ir += 1
		#   rd = facto_type.(b - A * xr)
		#   xr2_F16 = similar(b)
		#   normalize_vect!(xr2_F16, rd, D, model.nls_meta.nequ, model.meta.nvar)
		#   # print("\n", norm(xr2_F16))
		#   ldl_solve!(model.nls_meta.nequ + model.meta.nvar, xr2_F16, LDLT.L.colptr, LDLT.L.rowval, LDLT.L.nzval, LDLT.D, P)
		#   # print("\n", norm(xr2_F16))
		#   xr2 = convert(Array{T, 1}, xr2_F16)
		#   xr += xr2
	  #   end
	  # end


	  δr = xr[1 : model.nls_meta.nequ]
      δ = xr[model.nls_meta.nequ + 1 : end]
      δr2 = norm(δr)^2 / 2

      # We denormalize δ
	  if normalize != :None && facto_type != Float16
        denormalize_vect!(δ, col_norms, model.meta.nvar)
      end
	  if normalize == :A
		δ /= sqrt(λ)
	  end
    end

	if T != facto_type
	  δ = convert(Array{T, 1}, δ)
	end

    # Check model decrease
    # if δr2 > obj
    #   @error "‖δr‖² > ‖r‖²" δr2 obj
    #   fail = true
    #   continue
    # end

    x_suiv .=  x + δ
    residual!(model, x_suiv, r_suiv)
    norm_rsuiv = norm(r_suiv)
    obj_suiv = norm_rsuiv^2 / 2

    # Step not accepted : d(||r||²) < 1e-4 (||Jδ + r||² - ||r||²)
    pred = obj - δr2       # predicted reduction
    ared = obj - obj_suiv  # actual reduction
	step_accepted = ared ≥ 1e-4 * pred
    step_accepted_str = (step_accepted && δr2 <= obj) ? "acc" : "rej"

    ntimes = 0
    # Linear search along the δ direction
    if linesearch
      while !step_accepted && ntimes < 4
        δ /= δd
        x_suiv .=  x + δ
        residual!(model, x_suiv, r_suiv)
        norm_rsuiv = norm(r_suiv)
        obj_suiv = norm_rsuiv^2 / 2
        if facto == :QR
          # δr2 = ‖Jδ + r‖²
          δr2 = norm(A[1 : model.nls_meta.nequ, :] * δ + r)^2 / 2
        elseif facto == :LDL
          # δrₖ₊₁ = (δrₖ + r) / δd - r = (δrₖ - r) / δd
          # δrₖ = - Jδₖ - r
          δr = (δr - r) / δd
          δr2 = norm(δr)^2 / 2
        end

        # # Check model decrease
        # if δr2 > obj
        #   @error "‖δr‖² > ‖r‖²" δr2 obj
        #   fail = true
        #   continue
        # end

        # Step not accepted : d(||r||²) < 1e-4 (||Jδ + r||² - ||r||²)
        pred = obj - δr2       # predicted reduction
        ared = obj - obj_suiv  # actual reduction
        step_accepted = ared ≥ 1e-4 * pred
        step_accepted_str = (step_accepted && δr2 <= obj) ? "acc" : "rej"
        ntimes += 1
      end
    end

    norm_δ = norm(δ)
	if isnan(norm_δ)
	  @error "‖δ‖ = NaN"
	  fail2 = true
	  continue
	end

    @info log_row((iter, obj, old_obj - obj, norm_Jtr, λ, norm_δ, ared / pred, step_accepted_str))

    if !step_accepted
      # Update λ
      λ = max(λ, 1 / norm_δ) * νm^(ntimes + 1)

      # Update A
      if facto == :QR
        vals_A[model.nls_meta.nnzj + 1 : end] .= sqrt(λ)
        A = sparse(rows_A, cols_A, vals_A)

      # Update A
      elseif facto == :LDL
        if normalize == :A
          b[1 : model.nls_meta.nequ] .= - sqrt(λ) * r
          vals_A[1 : model.nls_meta.nequ] .= sqrt(λ)
          vals_A[model.nls_meta.nequ + model.nls_meta.nnzj + 1 : end] .= -sqrt(λ)
        else
          vals_A[model.nls_meta.nequ + model.nls_meta.nnzj + 1 : end] .= -λ
        end
        A = sparse(rows_A, cols_A, vals_A)
      end

    else
      # Update λ and x
      if ntimes > 0
        λ /= νd^(ntimes - 1)
      else
        λ /= νd
      end
      if ared ≥ 0.9 * pred  # very successful step
        λ /= νd
      end
      λ = max(1.0e-8, λ)
      x .= x_suiv

      # Update J
      jac_coord_residual!(model, x, vals)

      # Update r and b
      old_obj = obj
      r .= r_suiv
      b[1 : model.nls_meta.nequ] .= -r
      norm_r = norm_rsuiv
      obj = obj_suiv

      # Update A and Jtr
      if facto == :QR
        @views vals_A[1 : model.nls_meta.nnzj] = vals
        vals_A[model.nls_meta.nnzj + 1 : end] .= sqrt(λ)
        A = sparse(rows_A, cols_A, vals_A)

        mul_sparse!(Jtr, cols, rows, vals, r, model.nls_meta.nnzj)

      # Update A and Jtr
      elseif facto == :LDL
        @views vals_A[model.nls_meta.nequ + 1 :  model.nls_meta.nequ + model.nls_meta.nnzj] = vals
        if normalize == :A
          b[1 : model.nls_meta.nequ] *= sqrt(λ)
          vals_A[1 : model.nls_meta.nequ] .= sqrt(λ)
          vals_A[model.nls_meta.nequ + model.nls_meta.nnzj + 1 : end] .= -sqrt(λ)
        else
          vals_A[model.nls_meta.nequ + model.nls_meta.nnzj + 1 : end] .= -λ
        end
        A = sparse(rows_A, cols_A, vals_A)

        mul_sparse!(Jtr, cols_J, rows, vals, r, model.nls_meta.nnzj)
      end

      # Update the stopping criteria
      norm_Jtr = norm(Jtr)
      small_step = norm_δ < satol + srtol * norm(x)
      first_order = norm_Jtr < ϵ_first_order
      small_residual = norm_r < restol
      # ‖rₖ₋₁‖²/2 - ‖rₖ‖²/2 < oatol + ortol * ‖rₖ₋₁‖²/2
      small_obj_change =  old_obj - obj < oatol + ortol * old_obj
    end

    tired = iter > ite_max || elapsed_time > max_time
	if facto_type == Float64
	  t_iter = time() - t0_iter
	  @info "\nTime iter: " t_iter
	end
  end

  # fail || (@info log_row((iter, obj, old_obj - obj, norm_Jtr, λ, norm_δ, δr2, δr2 - obj, "")))

  if small_step
    status = :small_step
  elseif first_order
    status = :first_order
  elseif small_residual
    status = :small_residual
  elseif small_obj_change
    status = :acceptable
  elseif fail
    status = :neg_pred
  elseif fail2
	status = :exception
  elseif tired
    status = :max_iter
  end

  elapsed_time = time() - start_time

  stats = GenericExecutionStats(status,
                               model,
                               solution = x,
                               objective = obj,
                               iter = iter,
                               elapsed_time = elapsed_time,
                               dual_feas = norm_Jtr)
  @info stats
  return stats
end
