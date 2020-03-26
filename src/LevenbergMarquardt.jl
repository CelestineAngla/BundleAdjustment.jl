using LinearAlgebra


"""
Implementation of Levenberg Marquardt algorithm for least square problems
Where β₀ is the initial guess, residual is (f(x,β) - y),
jacob is the Jacobian of f, epsilon the stopping criteria,
and X and Y the values of the dataset.
"""
function LevenbergMarquardt(beta_0::Array{Float64,1}, residual, jacob, epsilon::Float64, X::Array{Float64,1}, Y::Array{Float64,1})::Array{Float64,1}
  print("\n-------- \n \n")
  lambda = 0.1 # regularization coefficient
  beta = beta_0
  diff = [10.0+epsilon 10.0+epsilon]
  ite = 0
  J = jacob(X, beta_0)
  # B = J
  r = residual(X, Y, beta_0)
  while norm(diff) > epsilon
    print("Iteration: ", ite, " Objective: ", norm(r)^2, "\n")
    Jt = transpose(J)
    JtJ = Jt*J

    # # Solve (JᵗJ + λ×Diag(JᵗJ))p = - Jᵗr  with Cholesky factorization
    # L = Cholesky(JtJ + lambda*Diagonal(JtJ))
    # inv_L = inv(L)
    # diff = transpose(inv_L)*inv_L*Jt*r

    # Solve min ||[J √λD]p + [r 0]||² with QR factorization
    D = Matrix{Float64}(I, size(JtJ)[1], size(JtJ)[1])
    # D = sqrt.(Diagonal(JtJ))
    A = [J; sqrt(lambda)*D]
    b = [r; zeros(size(A)[1]-length(r))]
    # Q, R = QR_Householder(A)
    # diff = Solve_triangle(R, transpose(Q)*b)
    diff = A \ b

    beta -= diff
    r_suiv = residual(X, Y, beta)
    # Step not accepted
    if LinearAlgebra.norm(r_suiv) > LinearAlgebra.norm(r)
      lambda *= 1.5
      beta += diff
    #Step accepted
    else
      lambda \= 5
      J = jacob(X, beta)
      # print("\nJ\n")
      # print(J)
      # B = Broyden(B, r_suiv-r, -diff)
      # print("\nB\n")
      # print(B)
      r = r_suiv
    end
    ite += 1
  end
  print("\nNumber of iterations: ", ite, "\n")
  return beta
end


function Levenberg_Marquardt(model::AbstractNLSModel, beta_0::Array{Float64,1}, epsilon::Float64)
  print("\n-------- \n \n")
  lambda = 0.1 # regularization coefficient
  beta = beta_0
  diff = fill(10.0+epsilon, model.meta.nvar)
  ite = 0
  J = jac_residual(model, beta_0)
  r = residual(model, beta_0)
  while norm(diff) > epsilon
    print("Iteration: ", ite, " Objective: ", 0.5*norm(r)^2, "\n")
    # Solve min ||[J √λD]p + [r 0]||² with QR factorization
    D = Matrix{Float64}(I, model.meta.nvar, model.meta.nvar)
    A = [J; sqrt(lambda)*D]
    b = [r; zeros(size(A)[1]-length(r))]
    diff = A \ b

    beta -= diff
    r_suiv = residual(model, beta)
    # Step not accepted
    if LinearAlgebra.norm(r_suiv) > LinearAlgebra.norm(r)
      lambda *= 1.5
      beta += diff
    #Step accepted
    else
      lambda \= 5
      J = jac_residual(model, beta)
      r = r_suiv
    end
    ite += 1
  end
  print("\nNumber of iterations: ", ite, "\n")
  return beta
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

"""
Computes the function e^(aX) + bX - Y
"""
function exponential(X::Array{Float64,1}, Y::Array{Float64,1}, par::Array{Float64,1})::Array{Float64,1}
  a, b = par
  return exp.(a*X) + b*X - Y
end


"""
Jacobian of the exponential function
(Xe^(aX) X)
"""
function jacobian(X::Array{Float64,1}, par::Array{Float64,1})::Array{Float64,2}
  n = length(X)
  a, b = par
  Ja = X.*exp.(a*X)
  Jb = X
  return [Ja Jb]
end


"""
Least square function
||e^(aX) + bX - Y||
"""
function least_square(X::Array{Float64,1}, Y::Array{Float64,1}, par::Array{Float64,1})::Float64
  return norm(exponential(X, Y, par))
end

#
# x_data = [2.2, 2.0, 1.9, 1.8, 1.28, 1.33, 1.12, 1.1, 0.8, 0.5, 1.7, 1.5, -14.8, -14.0, -12.0, -1.5, 1.0, 0.0, -1.0, -2.0, -5.0, -3.0, -4.0, -10.0, -15.0, -6.0, -4.50]
# y_data = [88.0, 60.6, 50.4, 42.0, 16.7758, 18.286, 12.75, 12.33, 7.35, 4.22, 35.0, 14.62, -44.4, -42.0, -35.9, -4.4502, 10.4, 0.99, -2.87, -5.98, -15.0, -8.9, -11.9996, -30.0, -43.0, -17.99999, -13.499877]
# guess_abs = [[0.1 15.7]; [5.5 -10.5]; [100.1 100.0]; [8.9 5.0]]
#
# for  i = 1:size(guess_abs)[1]
#   guess_ab = guess_abs[i,:]
#   epsilon = 0.0001
#   a, b = @time LevenbergMarquardt(guess_ab, exponential, jacobian, epsilon, x_data, y_data)
#   print("Intial guess: ", guess_ab)
#   print("\nLM results: ", a, " ", b, " \n ")
# end
