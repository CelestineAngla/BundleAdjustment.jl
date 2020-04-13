using LinearAlgebra
using SparseArrays
using SolverTools
using NLPModels
using LDLFactorizations
include("LMA_aux.jl")



"""
Implementation of Levenberg Marquardt algorithm for NLSModels
Solves min 1/2 ||r(x)||² where r is a vector of residuals
"""
function Levenberg_Marquardt(model :: AbstractNLSModel,
							 facto :: Symbol,
							 x :: AbstractVector=copy(model.meta.x0),
							 restol :: Float64=1e-3,
							 satol :: Float64=1e-3, srtol :: Float64=1e-3,
							 otol :: Float64=1e-3,
							 atol :: Float64=1e-3, rtol :: Float64=1e-3,
							 νd :: Float64=3.0, νm :: Float64=3.0, λ :: Float64=1.5,
							 ite_max :: Int=100)

  start_time = time()
  elapsed_time = 0.0
  iter = 0
  step_accepted = ""
  δ = 0
  T = eltype(x)
  x_suiv = Vector{T}(undef, length(x))


  if facto == :QR

	  # Initialize residuals
	  print("\n r")
	  r = @time residual(model, x)
	  sq_norm_r = norm(r)^2
	  r_suiv = copy(r)
	  # Initialize b = [r; 0]
	  b = [r; zeros(model.meta.nvar)]
	  xr = similar(b)

	  # Initialize J in the format J[rows[k], cols[k]] = vals[k]
	  print("\n J")
	  rows = Vector{Int}(undef, model.nls_meta.nnzj)
	  cols = Vector{Int}(undef, model.nls_meta.nnzj)
	  @time jac_structure_residual!(model, rows, cols)
	  @time vals = jac_coord_residual(model, x)
	  # Initialize A = [J; √λI] as a sparse matrix
	  print("\n A")
	  @time A = sparse(vcat(rows,collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(cols, collect(1 : model.meta.nvar)), vcat(vals, fill(sqrt(λ), model.meta.nvar)), model.nls_meta.nequ + model.meta.nvar, model.meta.nvar)

	  # Variables used in the stopping criteria
	  Jtr = transpose(A[1 : model.nls_meta.nequ, :])*r
	  ϵ_first_order = atol + rtol * norm(Jtr)
	  old_obj = 0.5 * sq_norm_r

	  # Stopping criteria
	  small_step = false
	  first_order = norm(Jtr) < ϵ_first_order
	  small_residual = norm(r) < restol
	  small_obj_change =  false
	  tired = iter > ite_max
	  status = :unknown

	  @info log_header([:iter, :f, :dual, :step, :bk], [Int, T, T, T, String],
	  hdr_override=Dict(:f=>"objective", :dual=>"objective reduction", :step=>"step norm", :bk=>"step accepted"))

	  while !(small_step || first_order || small_residual || small_obj_change || tired)

		@info log_row([iter, 0.5 * sq_norm_r, old_obj - 0.5 * sq_norm_r, norm(δ), step_accepted])

	    # Solve min ||[J √λI] δ + [r 0]||² with QR factorization
	    print("\nδ ")
		@time begin
		δ, δr = solve_qr!(xr, A, b)
		end
	    x_suiv .=  x - δ
	    @time residual!(model, x_suiv, r_suiv)
		iter += 1

	    # Step not accepted : d(||r||²) > 1e-4 (||Jδ + r||² - ||r||²)
	    if norm(r_suiv)^2 - sq_norm_r >= 1e-4 * (norm(δr)^2 - sq_norm_r)
		  step_accepted = "false"
	      # Update λ and A
	      λ *= νm
	      A[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar, :] *= sqrt(νm)

	    #Step accepted
	    else
		  step_accepted = "true"
	      # Update λ and x
	      λ /= νd
	      x .= x_suiv

	      # Update J and A
	      print("\nJ ")
	      @time jac_coord_residual!(model, x, vals)
	      print("\nA ")
	      @time A = sparse(vcat(rows,collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(cols, collect(1 : model.meta.nvar)), vcat(vals, fill(sqrt(λ), model.meta.nvar)), model.nls_meta.nequ + model.meta.nvar, model.meta.nvar)

		  # Update r and b
		  old_obj = 0.5*sq_norm_r
	      r .= r_suiv
	      sq_norm_r = norm(r)^2
	      b[1 : model.nls_meta.nequ] .= r

		  # Update Jtr
	      mul!(Jtr, transpose(A[1 : model.nls_meta.nequ, :] ), r)

		  # Update the stopping criteria
		  small_step = norm(δ) < satol + srtol * norm(x)
	      first_order = norm(Jtr) < ϵ_first_order
	      small_residual = norm(r) < restol
	      small_obj_change =  old_obj - 0.5 * sq_norm_r < otol * old_obj
	      tired = iter > ite_max
	    end
	  end


  elseif facto == :LDL

	  # Initialize residuals
	  print("\n r")
	  r = @time residual(model, x)
	  sq_norm_r = norm(r)^2
	  r_suiv = copy(r)
	  # Initialize b = [r; 0]
	  b = [r; zeros(model.meta.nvar)]

	  # Initialize J in the format J[rows[k], cols[k]] = vals[k]
	  print("\n J")
	  rows = Vector{Int}(undef, model.nls_meta.nnzj)
	  cols = Vector{Int}(undef, model.nls_meta.nnzj)
	  @time jac_structure_residual!(model, rows, cols)
	  @time vals = jac_coord_residual(model, x)
	  # Initialize A = [[I J]; [Jᵀ - λI]] as sparse upper-triangular matrix
	  cols .+= model.nls_meta.nequ
	  A = sparse(vcat(collect(1 : model.nls_meta.nequ), rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(collect(1 : model.nls_meta.nequ), cols, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(fill(1.0, model.nls_meta.nequ), vals, fill(-λ, model.meta.nvar)))

	  # Variables used in the stopping criteria
	  Jtr = transpose(A[1 : model.nls_meta.nequ, model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar])*r
	  ϵ_first_order = atol + rtol * norm(Jtr)
	  old_obj = 0.5 * sq_norm_r

	  # Stopping criteria
	  small_step = false
	  first_order = norm(Jtr) < ϵ_first_order
	  small_residual = norm(r) < restol
	  small_obj_change =  false
	  tired = iter > ite_max
	  status = :unknown

	  @info log_header([:iter, :f, :dual, :step, :bk], [Int, T, T, T, String],
	  hdr_override=Dict(:f=>"objective", :dual=>"objective reduction", :step=>"step norm", :bk=>"step accepted"))

	  while !(small_step || first_order || small_residual || small_obj_change || tired)

		@info log_row([iter, 0.5 * sq_norm_r, old_obj - 0.5 * sq_norm_r, norm(δ), step_accepted])

	    # Solve [[I J]; [Jᵀ - λI]] X = [r; 0] with LDL factorization
	    print("\nδ ")
		@time begin
		LDLT = ldl(A)
		X = LDLT \ b
		δr = X[1 : model.nls_meta.nequ]
		δ = X[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar]
		end
	    x_suiv .=  x - δ
	    @time residual!(model, x_suiv, r_suiv)
		iter += 1

	    # Step not accepted : d(||r||²) > 1e-4 (||Jδ + r||² - ||r||²)
	    if norm(r_suiv)^2 - sq_norm_r >= 1e-4 * (norm(δr)^2 - sq_norm_r)
		  step_accepted = "false"
	      # Update λ and A
	      λ *= νm
	      A[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar, model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar] *= sqrt(νm)

	    #Step accepted
	    else
		  step_accepted = "true"
	      # Update λ and x
	      λ /= νd
	      x .= x_suiv

	      # Update J and A
	      print("\nJ ")
	      @time jac_coord_residual!(model, x, vals)
	      print("\nA ")
	      @time A = sparse(vcat(collect(1 : model.nls_meta.nequ), rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(collect(1 : model.nls_meta.nequ), cols, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(fill(1.0, model.nls_meta.nequ), vals, fill(-λ, model.meta.nvar)))

		  # Update r and b
		  old_obj = 0.5*sq_norm_r
	      r .= r_suiv
	      sq_norm_r = norm(r)^2
	      b[1 : model.nls_meta.nequ] .= r

		  # Update Jtr
	      mul!(Jtr, transpose(A[1 : model.nls_meta.nequ, model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar]), r)

		  # Update the stopping criteria
		  small_step = norm(δ) < satol + srtol * norm(x)
	      first_order = norm(Jtr) < ϵ_first_order
	      small_residual = norm(r) < restol
	      small_obj_change =  old_obj - 0.5 * sq_norm_r < otol * old_obj
	      tired = iter > ite_max
	    end
	  end
  end



  @info log_row(Any[iter, 0.5 * sq_norm_r, old_obj - 0.5 * sq_norm_r, norm(δ), step_accepted])

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

  return GenericExecutionStats(status, model, solution=x, objective=0.5*sq_norm_r, iter=iter, elapsed_time=elapsed_time)
end
