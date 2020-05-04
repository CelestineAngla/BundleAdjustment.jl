using SparseArrays
using SolverTools
using NLPModels
using AMD
using Metis
include("lma_aux.jl")
include("ldl_aux.jl")
include("qr_aux.jl")



"""
Implementation of Levenberg Marquardt algorithm for NLSModels
Solves min 1/2 ||r(x)||² where r is a vector of residuals
"""
function Levenberg_Marquardt(model :: AbstractNLSModel,
							 facto :: Symbol,
							 perm :: Symbol;
							 x :: AbstractVector=copy(model.meta.x0),
							 restol=sqrt(eps(eltype(x))),
							 satol=sqrt(eps(eltype(x))), srtol=sqrt(eps(eltype(x))),
							 otol=sqrt(eps(eltype(x))),
							 atol=sqrt(eps(eltype(x))), rtol=sqrt(eps(eltype(x))),
							 νd :: Real=3.0, νm :: Real=3.0, λ :: Real=1.5,
							 ite_max :: Int=100)

  start_time = time()
  elapsed_time = 0.0
  iter = 0
  step_accepted = ""
  δ = 0
  T = eltype(x)
  νd = convert(T, νd)
  νm = convert(T, νm)
  λ = convert(T, λ)
  x_suiv = Vector{T}(undef, length(x))


  # Initialize residuals
  r = residual(model, x)
  sq_norm_r = norm(r)^2
  sq_norm_r₀ = sq_norm_r
  r_suiv = copy(r)

  # Initialize J in the format J[rows[k], cols[k]] = vals[k]
  rows = Vector{Int}(undef, model.nls_meta.nnzj)
  cols = Vector{Int}(undef, model.nls_meta.nnzj)
  jac_structure_residual!(model, rows, cols)
  vals = jac_coord_residual(model, x)
  old_vals = similar(vals)

  # Nearly zero or nearly linear residuals
  resatol = 100 * restol
  resrtol = 100 * restol
  snd_order = false
  counter_res_null = 0
  p = 4
  A_reglin = hcat(ones(Int64, p), collect(1 : p))
  b_reglin = Vector{T}(undef, p)


  if facto == :QR

	  # Initialize b = [r; 0]
	  b = [r; zeros(T, model.meta.nvar)]
	  xr = similar(b)

	  # Initialize A = [J; √λI] as a sparse matrix
	  A = sparse(vcat(rows,collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(cols, collect(1 : model.meta.nvar)), vcat(vals, fill(sqrt(λ), model.meta.nvar)), model.nls_meta.nequ + model.meta.nvar, model.meta.nvar)
	  col_norms = Vector{T}(undef, model.meta.nvar)
	  # QR_J = qr(A[1 : model.nls_meta.nequ, :])

	  Jtr = transpose(A[1 : model.nls_meta.nequ, :])*r

  elseif facto == :LDL

	  # Initialize b = [-r; 0]
	  b = [-r; zeros(T, model.meta.nvar)]
	  xr = similar(b)

	  # Initialize A = [[I J]; [Jᵀ - λI]] as sparse upper-triangular matrix
	  cols .+= model.nls_meta.nequ
	  A = sparse(vcat(collect(1 : model.nls_meta.nequ), rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(collect(1 : model.nls_meta.nequ), cols, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(fill(1.0, model.nls_meta.nequ), vals, fill(-λ, model.meta.nvar)))
	  col_norms = Vector{T}(undef, model.meta.nvar + model.nls_meta.nequ)
	  if perm == :AMD
		  P = amd(A)
	  elseif perm == :Metis
		  P , _ = Metis.permutation(A' + A)
	  	  P = convert(Array{Int64,1}, P)
	  end
	  Cp, Ci, Lp, parent, Lnz, Li, Lx, D, Y, pattern, flag, pinv = ldl_symbolic_aux(A, P, model.meta.nvar + model.nls_meta.nequ)
	  Jtr = transpose(A[1 : model.nls_meta.nequ, model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar])*r
  end

  ϵ_first_order = atol + rtol * norm(Jtr)
  old_obj = sq_norm_r / 2

  # Stopping criteria
  small_step = false
  first_order = norm(Jtr) < ϵ_first_order
  small_residual = norm(r) < restol
  small_obj_change =  false
  tired = iter > ite_max
  status = :unknown

  @info log_header([:iter, :f, :dual, :step, :bk], [Int, T, T, T, String],
  hdr_override=Dict(:f=>"½‖r‖² ", :dual=>" d(½‖r‖²)", :step=>"‖δ‖  ", :bk=>"step accepted"))

  while !(small_step || first_order || small_residual || small_obj_change || tired)

	@info log_row([iter, sq_norm_r / 2, old_obj - sq_norm_r / 2, norm(δ), step_accepted])

	# If the residuals are nearly zeros 5 times in a row, we use 2nd order derivatives
	# if sq_norm_r > resatol + resrtol * sq_norm_r₀
	# 	counter_res_null += 1
	# else
	# 	counter_res_null = 0
	# end
	# if counter_res_null > 5
	# 	snd_order = true
	# end
	#
	# fill_b_reglin!(b_reglin, p, iter, sq_norm_r)
	# if iter >= p - 1
	# 	x_reglin = A_reglin \ b_reglin
	# 	print("\nx\n", x_reglin)
	# end
	#
	# if jac_not_const(old_vals, vals, model.meta.nnzj)
	# 	snd_order = true
	# end




	if facto == :QR
	    # Solve min ||[J √λI] δ + [r 0]||² with QR factorization

		# G_list = Vector{LinearAlgebra.Givens{Float64}}(undef, Int(model.meta.nvar*(model.meta.nvar + 1)/2))
		# news = Vector{Float64}(undef, model.meta.nvar)
		# counter = fullQR_givens!(QR_J.R, G_list, news, sqrt(λ), model.meta.nvar, model.nls_meta.nequ)
		# Prow = vcat(QR_J.prow, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
		# Pcol = vcat(QR_J.pcol, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
		# δ, δr = solve_qr2!(model.nls_meta.nequ + model.meta.nvar, model.meta.nvar, xr, b, QR_J.Q, QR_J.R, Prow, Pcol, counter, G_list)

		normalize_cols!(A, col_norms, model.meta.nvar)
		if perm == :AMD
			QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_AMD)
		elseif perm == :Metis
			QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_METIS)
		end

		δ, δr = solve_qr!(model.nls_meta.nequ + model.meta.nvar, model.meta.nvar, xr, b, QR.Q, QR.R, QR.prow, QR.pcol)
		denormalize!(δ, col_norms, model.meta.nvar)
		denormalize!(δr, col_norms, model.meta.nvar)
		x_suiv .=  x - δ

	elseif facto == :LDL
		# Solve [[I J]; [Jᵀ - λI]] X = [r; 0] with LDL factorization
		normalize_cols!(A, col_norms, model.meta.nvar + model.nls_meta.nequ)
		LDLT = ldl(A, P, upper=true)
		LDLT = ldl_numeric_aux(A, P, model.meta.nvar + model.nls_meta.nequ, Cp, Ci, Lp, parent, Lnz, Li, Lx, D, Y, pattern, flag, pinv)
		xr .= b
		ldl_solve!(model.nls_meta.nequ + model.meta.nvar, xr, LDLT.L.colptr, LDLT.L.rowval, LDLT.L.nzval, LDLT.D, P)
		denormalize!(xr, col_norms, model.nls_meta.nequ + model.meta.nvar)
		δr = xr[1 : model.nls_meta.nequ]
		δ = xr[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar]
		x_suiv .=  x + δ
	end

	residual!(model, x_suiv, r_suiv)
	iter += 1

    # Step not accepted : d(||r||²) > 1e-4 (||Jδ + r||² - ||r||²)
    if norm(r_suiv)^2 - sq_norm_r >= 1e-4 * (norm(δr)^2 - sq_norm_r)
	  step_accepted = "false"
      # Update λ
      λ *= νm

	  # Update A
	  if facto == :QR
		  denormalize_cols!(A, col_norms, model.meta.nvar)
		  A[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar, :] *= sqrt(νm)
	  elseif facto == :LDL
		  denormalize_cols!(A, col_norms, model.meta.nvar + model.nls_meta.nequ)
		  A[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar, model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar] *= νm
	  end

    #Step accepted
    else
	  step_accepted = "true"

      # Update λ and x
      λ /= νd
      x .= x_suiv

	  # Update J and r
	  old_vals .= vals
	  jac_coord_residual!(model, x, vals)
	  old_obj = sq_norm_r / 2
      r .= r_suiv
      sq_norm_r = norm(r)^2

	  # Update A, b and Jtr
	  if facto == :QR
		  A = sparse(vcat(rows,collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(cols, collect(1 : model.meta.nvar)), vcat(vals, fill(sqrt(λ), model.meta.nvar)), model.nls_meta.nequ + model.meta.nvar, model.meta.nvar)
		  # QR_J = qr(A[1 : model.nls_meta.nequ, :])
		  b[1 : model.nls_meta.nequ] .= r
	      mul!(Jtr, transpose(A[1 : model.nls_meta.nequ, :] ), r)

	  elseif facto == :LDL
		  A = sparse(vcat(collect(1 : model.nls_meta.nequ), rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(collect(1 : model.nls_meta.nequ), cols, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(fill(1.0, model.nls_meta.nequ), vals, fill(-λ, model.meta.nvar)))
		  b[1 : model.nls_meta.nequ] .= -r
		  mul!(Jtr, transpose(A[1 : model.nls_meta.nequ, model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar]), r)
	  end

	  # Update the stopping criteria
	  small_step = norm(δ) < satol + srtol * norm(x)
      first_order = norm(Jtr) < ϵ_first_order
      small_residual = norm(r) < restol
      small_obj_change =  old_obj - sq_norm_r / 2 < otol * old_obj
    end

	tired = iter > ite_max
  end

  @info log_row(Any[iter, sq_norm_r / 2, old_obj - sq_norm_r / 2, norm(δ), step_accepted])

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

  return GenericExecutionStats(status, model, solution=x, objective=sq_norm_r/2, iter=iter, elapsed_time=elapsed_time, primal_feas=norm(Jtr))
end
