"""
Solves A x = b using the QR factorization of A
"""
function solve_qr!(xr, A, b)
  QR = qr(A)
  m = size(QR.Q, 1)
  n = size(QR.R, 2)
  m ≥ n || error("currently, this function only supports overdetermined problems")
  @assert length(b) == m
  @assert length(xr) == m

  # SuiteSparseQR decomposes P₁ * A * P₂ = Q * R, where
  # * P₁ is a permutation stored in QR.prow;
  # * P₂ is a permutation stored in QR.pcol;
  # * Q  is orthogonal and stored in QR.Q;
  # * R  is upper trapezoidal and stored in QR.R.
  #
  # The solution of min ‖Ax - b‖ is thus given by
  # x = P₂ R⁻¹ Q' P₁ b.
  mul!(xr, QR.Q', b[QR.prow])  # xr ← Q'(P₁b)  NB: using @views here results in tons of allocations?!
  @views x = xr[1:n]
  ldiv!(LinearAlgebra.UpperTriangular(QR.R), x)  # x ← R⁻¹ x
  @views x[QR.pcol] .= x
  @views r = xr[n+1:m]  # = Q₂'b
  return x, r
end


# function solve_qr_reg!(mat, ncon, nvar)
# 	n = nvar
# 	m = ncon
# 	Qλ = Matrix{Float64}(I, m + n, m + n)
# 	Rλ = copy(mat)
# 	print("\n\n", Rλ)
# 	for k = 1 : n
# 		print("\n\n", k)
# 		l = n - k + 1
# 		G, r = givens(mat, l, m + l, l)
# 		Qλ = G * Qλ
# 		Rλ = G * mat
# 		print("\n\n", Rλ)
# 		for i = 1 : k - 1
# 			G, r = givens(mat, n - i + 1, m + n - k + 1, n - i + 1)
# 			Qλ = G * Qλ
# 			Rλ = G * mat
# 			print("\n\n", Rλ)
# 		end
# 	end
# 	return Qλ', Rλ
# end
#
# m = 7
# n = 5
# λ = 1.5
# A = rand([-5.0, 5.0], m, n)
# QR_A = qr(A)
# print("\n\n", QR_A.R)
# mat = [QR_A.R; zeros(m - n, n); sqrt(λ) * Matrix{Float64}(I, n, n)]
# Qλ, Rλ = solve_qr_reg!(mat, m, n)
#
# print("\n\n", Qλ*Rλ, "\n\n", mat)



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
