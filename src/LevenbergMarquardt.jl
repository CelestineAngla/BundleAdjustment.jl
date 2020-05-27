using SparseArrays
using SolverTools
using NLPModels
using AMD
using Metis
using UnicodePlots
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
							 normalize :: Symbol;
                             x :: AbstractVector=copy(model.meta.x0),
                             restol=100*sqrt(eps(eltype(x))),
                             satol=sqrt(eps(eltype(x))), srtol=sqrt(eps(eltype(x))),
                             oatol=0*sqrt(eps(eltype(x))), ortol=0*sqrt(eps(eltype(x))),
                             atol=100*sqrt(eps(eltype(x))), rtol=100*sqrt(eps(eltype(x))),
                             νd=eltype(x)(3), νm=eltype(x)(3), λ=eltype(x)(1.5), δd=eltype(x)(2),
                             ite_max :: Int=500)

  show(model)
  print("\n")

  start_time = time()
  elapsed_time = 0.0
  iter = 0
  x_suiv = similar(x)
  T = eltype(x)
  norm_δ = 0
  δr2 = 0
  step_accepted_str = ""

  # Initialize residuals
  r = residual(model, x)
  norm_r = norm(r)
  obj = norm_r^2 / 2
  sq_norm_r₀ = 2 * obj
  r_suiv = copy(r)
  # Initialize b = [r; 0]
  b = [-r; zeros(T, model.meta.nvar)]
  xr = similar(b)

  # Initialize J in the format J[rows[k], cols[k]] = vals[k]
  rows = Vector{Int}(undef, model.nls_meta.nnzj)
  cols = Vector{Int}(undef, model.nls_meta.nnzj)
  jac_structure_residual!(model, rows, cols)
  vals = jac_coord_residual(model, x)

  # Nearly zero or nearly linear residuals
  resatol = 100 * restol
  resrtol = 100 * restol
  snd_order = false
  counter_res_null = 0
  Jδ = Vector{T}(undef, model.nls_meta.nequ)
  Jδ_suiv = similar(Jδ)
  jtol = restol

  if facto == :QR
    # Initialize A = [J; √λI] as a sparse matrix
    rows_A = vcat(rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    cols_A = vcat(cols, collect(1 : model.meta.nvar))
    vals_A = vcat(vals, fill(sqrt(λ), model.meta.nvar))
    A = sparse(rows_A, cols_A, vals_A)

    if normalize != :None
      col_norms = Vector{T}(undef, model.meta.nvar)
    end

    # Givens version
    # col_norms = ones(model.meta.nvar)
    # col_norms = Vector{T}(undef, model.meta.nvar)
    # normalize_cols!(A[1 : model.nls_meta.nequ, :], col_norms, model.meta.nvar)
    # QR_J = qr(A[1 : model.nls_meta.nequ, :])
    # R = similar(QR_J.R)
    # G_list = Vector{LinearAlgebra.Givens{Float64}}(undef, Int(model.meta.nvar*(model.meta.nvar + 1)/2))
    # news = Vector{Float64}(undef, model.meta.nvar)
    # Prow = vcat(QR_J.prow, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    # denormalize_cols!(A[1 : model.nls_meta.nequ, :], col_norms, model.meta.nvar)

    # Compute Jᵀr
    Jtr = transpose(A[1 : model.nls_meta.nequ, :]) * r

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
      vals_A = vcat(fill(1.0, model.nls_meta.nequ), vals, fill(-λ, model.meta.nvar))
    end
    A = sparse(rows_A, cols_A, vals_A)

    # Compute permutation and perform symbolic analyse of A
    if perm == :AMD
      P = amd(A)
    elseif perm == :Metis
      P , _ = Metis.permutation(A' + A)
    end
    ldl_symbolic = ldl_analyse(A, P, upper=true, n=model.meta.nvar + model.nls_meta.nequ)

    # Compute Jᵀr
    Jtr = transpose(A[1 : model.nls_meta.nequ, model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar]) * r

    # Normalize J and multiply -λI by D²
    if normalize != :None
      col_norms = Vector{T}(undef, model.meta.nvar)
      normalize_ldl!(A, col_norms, model.meta.nvar, model.nls_meta.nequ)
    end
  end

  norm_Jtr = norm(Jtr)
  ϵ_first_order = atol + rtol * norm_Jtr
  old_obj = obj

  # Stopping criteria
  small_step = false
  first_order = norm_Jtr < ϵ_first_order
  small_residual = norm_r < restol
  small_obj_change =  false
  tired = iter > ite_max
  status = :unknown


  @info log_header([:iter, :f, :bk, :dual, :ratio, :step, :radius, :slope, :cgstatus], [Int, T, T, T, T, T, T, T, String],
  hdr_override=Dict(:f=>"f(x)", :bk=>"Δf", :dual=>"‖Jᵀr‖", :ratio=>"λ", :step=>"‖δ‖  ", :radius=>"½‖δr‖²", :slope=>"½‖δr‖² - f", :cgstatus=>"step accepted"))

  while !(small_step || first_order || small_residual || small_obj_change || tired)
    if iter == 0
      @info log_row([iter, obj, old_obj - obj, norm_Jtr, λ])
    else
      @info log_row([iter, obj, old_obj - obj, norm_Jtr, λ, norm_δ, δr2, δr2 - obj, step_accepted_str])
    end

    # If the residuals are nearly zeros 5 times in a row, we use 2nd order derivatives
    # if sq_norm_r > resatol + resrtol * sq_norm_r₀
    # 	counter_res_null += 1
    # else
    # 	counter_res_null = 0
    # end
    # if counter_res_null > 5
    # 	snd_order = true
    # end

    if facto == :QR
      # Solve min ‖[J √λI] δ + [r 0]‖² with QR factorization

      # Givens version
      # R .= QR_J.R
      # counter = fullQR_givens!(R, G_list, news, sqrt(λ), col_norms, model.meta.nvar, model.nls_meta.nequ)
      # δ, δr = solve_qr2!(model.nls_meta.nequ + model.meta.nvar, model.meta.nvar, xr, b, QR_J.Q, R, Prow, QR_J.pcol, counter, G_list)
      # denormalize!(δ, col_norms, model.meta.nvar)

      if normalize == :A
        normalize_qr_a!(A, col_norms, model.meta.nvar)
      elseif normalize == :J
        normalize_qr_j!(A, col_norms, model.meta.nvar)
      end

      # Perform QR factorization and solve the problem
      if perm == :AMD
        QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_AMD)
      elseif perm == :Metis
        QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_METIS)
      end
      δ, δr = solve_qr!(model.nls_meta.nequ + model.meta.nvar, model.meta.nvar, xr, b, QR.Q, QR.R, QR.prow, QR.pcol)

      if normalize != :None
        denormalize_vect!(δ, col_norms, model.meta.nvar)
      end

      # δr = ‖[J; √λI] δ + [r; 0]‖² - λ‖δ‖² = ‖[R; 0]δ + [r; 0]‖² - λ‖δ‖²
      if normalize != :None
        denormalize_qr!(A, col_norms, model.meta.nvar)
      end
      # Qr = QR.Q[:, 1 : model.nls_meta.nequ] * r
      # δr2 = norm(QR.R * δ + Qr[1 : model.meta.nvar])^2  + norm(Qr[model.meta.nvar + 1 : end])^2 - λ * norm(δ)^2
      δr2 = norm(A[1 : model.nls_meta.nequ, :] * δ + r)^2 / 2

    elseif facto == :LDL
      # Solve [[I J]; [Jᵀ - λI]] X = [-r; 0] with LDL factorization
      LDLT = ldl_factorize(A, ldl_symbolic, true)
      xr .= b
      ldl_solve!(model.nls_meta.nequ + model.meta.nvar, xr, LDLT.L.colptr, LDLT.L.rowval, LDLT.L.nzval, LDLT.D, P)
      δr = xr[1 : model.nls_meta.nequ]
      δ = xr[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar]
      δr2 = norm(δr)^2 / 2

      if normalize != :None
        denormalize_vect!(δ, col_norms, model.meta.nvar)
        if normalize == :A
          δ /= sqrt(λ)
        end
      end
    end

    x_suiv .=  x + δ
    residual!(model, x_suiv, r_suiv)
    norm_rsuiv = norm(r_suiv)
    obj_suiv = norm_rsuiv^2 / 2
    iter += 1

    # Step not accepted : d(||r||²) < 1e-4 (||Jδ + r||² - ||r||²)
    step_accepted = obj_suiv - obj < 1e-4 * (δr2 - obj)
    @assert δr2 < obj

    # Linear search along the δ direction
    ntimes = 0
    # while !step_accepted && ntimes < 5
    # 	δ /= δd
    # 	x_suiv .=  x + δ
    # 	residual!(model, x_suiv, r_suiv)
    # 	if facto == :QR
    # 	  # δr2 = ‖Jδ + r‖²
    # 	  δr2 = norm(A[1 : model.nls_meta.nequ, :] * δ + r)^2 / 2
    #     elseif facto == :LDL
    # 	  # δrₖ₊₁ = (δrₖ + r) / δd - r = (δrₖ - r) / δd
    # 	  δr = (δr - r) / δd
    # 	  δr2 = norm(δr)^2 / 2
    #     end
    # 	step_accepted = obj_suiv - obj < 1e-4 * (δr2 - obj)
    #   @assert δr2 < obj
    # 	ntimes += 1
    # end

    if !step_accepted
      # Update λ
      λ *= νm#^ntimes

      # Update A
      if facto == :QR
        vals_A[model.nls_meta.nnzj + 1 : model.nls_meta.nnzj + model.meta.nvar] .= sqrt(λ)
        A = sparse(rows_A, cols_A, vals_A)

      # Update A
      elseif facto == :LDL
        if normalize == :A
          b[1 : model.nls_meta.nequ] *= sqrt(νm)#^ntimes
          A[1 : model.nls_meta.nequ, 1 : model.nls_meta.nequ] *= sqrt(νm)#^ntimes
          A[model.nls_meta.nequ + 1 : end, model.nls_meta.nequ + 1 : end] *= sqrt(νm)#^ntimes
        else
          A[model.nls_meta.nequ + 1 : end, model.nls_meta.nequ + 1 : end] *= νm#^ntimes
        end
      end

    else
      # Update λ and x
      if ntimes > 0
        λ *= νd^(ntimes - 2)
      else
        λ /= νd
      end
      x .= x_suiv

      # Update J and check if the jacobian is constant
      if facto == :QR
        # mul_sparse!(Jδ, rows, cols, vals, δ, model.nls_meta.nnzj)
        jac_coord_residual!(model, x, vals)
        # if jac_not_const(Jδ, Jδ_suiv, rows, cols, vals, δ, model.nls_meta.nnzj, jtol)
        #   snd_order = true
        # end
      elseif facto == :LDL
        # mul_sparse!(Jδ, rows, cols_J, vals, δ, model.nls_meta.nnzj)
        jac_coord_residual!(model, x, vals)
        # if jac_not_const(Jδ, Jδ_suiv, rows, cols_J, vals, δ, model.nls_meta.nnzj, jtol)
        #   snd_order = true
        # end
      end

      # Update r and b
      old_obj = obj
      r .= r_suiv
      b[1 : model.nls_meta.nequ] .= -r
      norm_r = norm_rsuiv
      obj = obj_suiv

      # Update A and Jtr
      if facto == :QR
        @views vals_A[1 : model.nls_meta.nnzj] = vals
        vals_A[model.nls_meta.nnzj + 1 : model.nls_meta.nnzj + model.meta.nvar] .= sqrt(λ)
        A = sparse(rows_A, cols_A, vals_A)

        # Givens version
        # normalize_cols!(A[1 : model.nls_meta.nequ, :], col_norms, model.meta.nvar)
        # QR_J = qr(A[1 : model.nls_meta.nequ, :])
        # denormalize_cols!(A[1 : model.nls_meta.nequ, :], col_norms, model.meta.nvar)

        mul!(Jtr, transpose(A[1 : model.nls_meta.nequ, :]), r)

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

        mul!(Jtr, transpose(A[1 : model.nls_meta.nequ, model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar]), r)

        if normalize != :None
          normalize_ldl!(A, col_norms, model.meta.nvar, model.nls_meta.nequ)
        end
      end

      # Update the stopping criteria
      norm_Jtr = norm(Jtr)
      norm_δ = norm(δ)
      small_step = norm_δ < satol + srtol * norm(x)
      first_order = norm_Jtr < ϵ_first_order
      small_residual = norm_r < restol
      # ‖rₖ₋₁‖²/2 - ‖rₖ‖²/2 < oatol + ortol * ‖rₖ₋₁‖²/2
      small_obj_change =  old_obj - obj < oatol + ortol * old_obj
    end

    if step_accepted
      step_accepted_str = "true"
    else
      step_accepted_str = "false"
    end
    tired = iter > ite_max
  end

  @info log_row(Any[iter, obj, old_obj - obj, norm_Jtr, λ, norm_δ, δr2, δr2 - obj, step_accepted_str])

  if small_step
    status = :small_step
  elseif first_order
    status = :first_order
  elseif small_residual
    status = :small_residual
  elseif small_obj_change
    status = :acceptable
  else
    status = :max_iter
  end

  elapsed_time = time() - start_time

  return GenericExecutionStats(status, model, solution=x, objective=obj, iter=iter, elapsed_time=elapsed_time, primal_feas=norm_Jtr)
end
