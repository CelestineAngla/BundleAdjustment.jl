using LinearAlgebra


"""
Implementation of Levenberg Marquardt algorithm for NLSModels
"""
function Levenberg_Marquardt(model::AbstractNLSModel, x0::Array{Float64,1}, atol::Float64, rtol::Float64, ite_max::Int)
  lambda = 1.5 # regularization coefficient
  x = x0
  ite = 0

  print("\n r")
  # Initialize residuals
  r = @time residual(model, x0)
  r_suiv = copy(r)
  # Initialize b = [r; 0]
  b = @time [r; zeros(model.meta.nvar)]

  print("\n J")
  # Initialize J in the format J[rows[k], cols[k]] = vals[k]
  rows = Vector{Int}(undef, model.nls_meta.nnzj)
  cols = Vector{Int}(undef, model.nls_meta.nnzj)
  @time jac_structure_residual!(model, rows, cols)
  vals = @time jac_coord_residual(model, x)
  print("\n A")
  # Initialize A = [J; √λI] as a sparse matrix
  A = @time sparse(rows, cols, vals, model.nls_meta.nequ + model.meta.nvar, model.meta.nvar)
  A = @time fill_sparse!(A, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar), collect(1 : model.meta.nvar), fill(sqrt(lambda), model.meta.nvar))

  # The stopping criteria is: stop = norm(Jᵀr) > stop_inf = atol + rtol*stop(0)
  stop = norm(transpose(A[1 : model.nls_meta.nequ, :])*r)
  stop_inf = atol + rtol*stop
  old_cost = 10000
  while  stop > stop_inf && ite < ite_max
    print("\nIteration: ", ite, ", Objective: ", 0.5*norm(r)^2, ", Stopping criteria: ", stop_inf, " ", stop, ", Scipy stopping criteria: ", old_cost, " ", 0.0001*old_cost, "\n")

    # Solve min ||[J √λI] δ + [r 0]||² with QR factorization
    print("\ndelta ")
    delta = @time A \ b
    x -= delta
    r_suiv = residual!(model, x, r_suiv)

    # Step not accepted
    if norm(r_suiv) > norm(r)
      print("\n/!\\ step not accepted /!\\ \n")
      # Update λ and A
      lambda *= 3
      A[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar, :] *= sqrt(3)
      # Cancel the change on x
      x += delta

    #Step accepted
    else
      # Update λ, A, b and stop
      lambda /= 3
      print("\njac ")
      vals = @time jac_coord_residual!(model, x, vals)
      print("\nfill ")
      # jac_coord_residual!(model, x, A[1 : model.nls_meta.nequ, :].nzval)
      A[1 : model.nls_meta.nequ, :] = @time fill_sparse!(A[1 : model.nls_meta.nequ, :], rows, cols, vals)
      A[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar, :] /= sqrt(3)
      old_cost = 0.5*norm(r)^2
      r = @time copy(r_suiv)
      b[1 : model.nls_meta.nequ] .= r
      stop = norm(transpose(A[1 : model.nls_meta.nequ, :] )*r)
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


"""
Broyden's formula to update Jacobian
Jᵢ = Jᵢ₋₁ + (Δrᵢ - Jᵢ₋₁ Δθᵢ)/|Δθᵢ|² × Δθᵢᵗ
"""
function Broyden(J, diff_residual, diff_param)
  return J + ((diff_residual - J*diff_param)*transpose(diff_param))/(transpose(diff_param)*diff_param)
end


"""
Computes L in the Cholesky factorisation of A
"""
function Cholesky(A::Array{Float64,2})::Array{Float64,2}
  n = size(A)[1]
  L = zeros(Float64, n, n)
  L[1,1] = sqrt(A[1,1])
  for j = 2:n
    L[j,1] = A[1,j]/L[1,1]
  end
  for i = 2:n
    L[i,i] = sqrt(A[i,i] - sum(L[i,k]^2 for k = 1:i-1))
    for j = i+1:n
      L[j,i] = (A[i,j] - sum(L[i,k]*L[j,k] for k = 1:i-1))/L[i,i]
    end
  end
  return L
end



"""
Computes Q and R in the Householder QR factorization of A
"""
function QR_Householder(A)
    m, n = size(A)
    Q = Matrix{Float64}(I, m, m)
    R = A
    for i = 1:n
        x = R[i:m, i]
        e = zeros(length(x))
        e[1] = 1
        u = sign(x[1])*norm(x)*e + x
        v = u/norm(u)
        R[i:m, 1:n] -= 2*v*transpose(v)*R[i:m, 1:n]
        Q[1:m, i:m] -= Q[1:m, i:m]*2*v*transpose(v)
    end
    return Q,R
end


"""
Solves triangular system Rx = b (where R is upper triangular)
"""
function Solve_triangle(R, b)
  m, n = size(R)
  x = zeros(n)
  for i = n:-1:1
    if R[i,i] != 0
      if i == n
        x[i] = b[i]/R[i,i]
      else
        x[i] = (b[i] - sum(x[j]*R[i,j] for j = i+1:n))/R[i,i]
      end
    end
  end
  return x
end
