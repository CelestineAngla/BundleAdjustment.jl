using LinearAlgebra


"""
Implementation of Levenberg Marquardt algorithm for NLSModels
"""
function Levenberg_Marquardt(model::AbstractNLSModel, x0::Array{Float64,1}, atol::Float64, rtol::Float64, ite_max::Int)
  lambda = 1.5 # regularization coefficient
  x = x0
  x_suiv = Vector{Float64}(undef, length(x))
  ite = 0

  print("\n r")
  # Initialize residuals
  r = @time residual(model, x0)
  sq_norm_r = norm(r)^2
  r_suiv = copy(r)
  # Initialize b = [r; 0]
  b = [r; zeros(model.meta.nvar)]

  print("\n J")
  # Initialize J in the format J[rows[k], cols[k]] = vals[k]
  rows = Vector{Int}(undef, model.nls_meta.nnzj)
  cols = Vector{Int}(undef, model.nls_meta.nnzj)
  @time jac_structure_residual!(model, rows, cols)
  @time vals = jac_coord_residual(model, x)
  print("\n A")
  # Initialize A = [J; √λI] as a sparse matrix
  A = sparse(rows, cols, vals, model.nls_meta.nequ + model.meta.nvar, model.meta.nvar)
  @time A = fill_sparse!(A, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar), collect(1 : model.meta.nvar), fill(sqrt(lambda), model.meta.nvar))

  # The stopping criteria is: stop = norm(Jᵀr) > stop_inf = atol + rtol*stop(0)
  Jtr = transpose(A[1 : model.nls_meta.nequ, :])*r
  stop = norm(Jtr)
  stop_inf = atol + rtol*stop
  while  stop > stop_inf && ite < ite_max
    print("\nIteration: ", ite, ", Objective: ", 0.5*sq_norm_r,  ", Stopping criteria: ", stop_inf, " ", stop, "\n")

    # Solve min ||[J √λI] δ + [r 0]||² with QR factorization
    print("\ndelta ")
    @time delta = A \ b
    @time x_suiv .=  x - delta
    @time residual!(model, x_suiv, r_suiv)

    # Step not accepted
    if norm(r_suiv)^2 - sq_norm_r >= 1e-4 * (norm(A[1 : model.nls_meta.nequ, :]*delta + r)^2 - sq_norm_r)
      print("\n/!\\ step not accepted /!\\ \n")
      # Update λ and A
      lambda *= 2
      A[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar, :] .*= sqrt(3)

    #Step accepted
    else
      # Update λ, x, A, b and stop
      lambda /= 5
      @time x .= x_suiv
      print("\njac ")
      @time vals = jac_coord_residual!(model, x, vals)
      print("\nfill ")
      # jac_coord_residual!(model, x, A[1 : model.nls_meta.nequ, :].nzval)
      @time A[1 : model.nls_meta.nequ, :] = fill_sparse!(A[1 : model.nls_meta.nequ, :], rows, cols, vals)
      @time A[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar, :] /= sqrt(3)
      @time r .= r_suiv
      @time sq_norm_r = norm(r)^2
      @time b[1 : model.nls_meta.nequ] .= r
      mul!(Jtr, transpose(A[1 : model.nls_meta.nequ, :] ), r)
      stop = norm(Jtr)
    end
    ite += 1
  end
  print("\nNumber of iterations: ", ite, "\n")
  return x
end

"""
Update the values of a sparse matrix
"""
function fill_sparse!(A, rows, cols, vals)
  for k = 1 : length(rows)
    A[rows[k], cols[k]] = vals[k]
  end
  return A
end
