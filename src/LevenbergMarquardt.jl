using LinearAlgebra


"""
Implementation of Levenberg Marquardt algorithm for NLSModels
"""
function Levenberg_Marquardt(model::AbstractNLSModel, x0::Array{Float64,1}, atol::Float64, rtol::Float64)
  lambda = 0.1 # regularization coefficient
  x = x0
  diff = fill(0.0, model.meta.nvar)
  ite = 0
  J = jac_residual(model, x0)
  r = residual(model, x0)
  stop = norm(transpose(J)*r)
  stop_inf = atol + rtol*stop
  D = Matrix{Float64}(I, model.meta.nvar, model.meta.nvar)
  while  stop > stop_inf
    print("Iteration: ", ite, " Objective: ", 0.5*norm(r)^2, ", Stopping criteria: ", stop, "\n")
    # Solve min ||[J √λD]p + [r 0]||² with QR factorization
    A = [J; sqrt(lambda)*D]
    b = [r; zeros(size(A)[1]-length(r))]
    diff = A \ b

    x -= diff
    r_suiv = residual(model, x)
    # Step not accepted
    if LinearAlgebra.norm(r_suiv) > LinearAlgebra.norm(r)
      print("\n/!\\ step not accepted /!\\ \n")
      lambda *= 3
      x += diff
    #Step accepted
    else
      lambda \= 3
      J = jac_residual(model, x)
      r = r_suiv
      stop = norm(transpose(J)*r)
    end
    ite += 1
  end
  print("\nNumber of iterations: ", ite, "\n")
  return x
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
