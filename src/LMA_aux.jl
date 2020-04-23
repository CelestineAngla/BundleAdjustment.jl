using LDLFactorizations
using LinearAlgebra

"""
Solves A x = b using the LDL factorization of A (A is symetric)
"""
function ldl_solve!(n, b, Lp, Li, Lx, D, P)
  @views y = b[P]
  ldl_lsolve!(n, y, Lp, Li, Lx)
  ldl_dsolve!(n, y, D)
  ldl_ltsolve!(n, y, Lp, Li, Lx)
  @views b[P] = y
  return b
end


function ldl_lsolve!(n, x, Lp, Li, Lx)
  @inbounds for j = 1:n
    xj = x[j]
    @inbounds for p = Lp[j] : (Lp[j+1] - 1)
      x[Li[p]] -= Lx[p] * xj
    end
  end
  return x
end


function ldl_dsolve!(n, x, D)
  @inbounds for j = 1:n
    x[j] /= D[j]
  end
  return x
end


function ldl_ltsolve!(n, x, Lp, Li, Lx)
  @inbounds for j = n:-1:1
    xj = x[j]
    @inbounds for p = Lp[j] : (Lp[j+1] - 1)
      xj -= Lx[p] * x[Li[p]]
    end
    x[j] = xj
  end
  return x
end



using SparseArrays: SparseMatrixCSC
import SuiteSparse.SPQR: CHOLMOD, _default_tol, _qr!, QRSparse
import SuiteSparse.CHOLMOD: Sparse
using SuiteSparse

function myqr(A::SparseMatrixCSC{Tv}; tol = _default_tol(A), ordering=SuiteSparse.SPQR.ORDERING_DEFAULT) where {Tv <: CHOLMOD.VTypes}
    R     = Ref{Ptr{CHOLMOD.C_Sparse{Tv}}}()
    E     = Ref{Ptr{CHOLMOD.SuiteSparse_long}}()
    H     = Ref{Ptr{CHOLMOD.C_Sparse{Tv}}}()
    HPinv = Ref{Ptr{CHOLMOD.SuiteSparse_long}}()
    HTau  = Ref{Ptr{CHOLMOD.C_Dense{Tv}}}(C_NULL)

    # SPQR doesn't accept symmetric matrices so we explicitly set the stype
    r, p, hpinv = _qr!(ordering, tol, 0, 0, Sparse(A, 0),
        C_NULL, C_NULL, C_NULL, C_NULL,
        R, E, H, HPinv, HTau)

    R_ = SparseMatrixCSC(Sparse(R[]))
    return QRSparse(SparseMatrixCSC(Sparse(H[])),
                    vec(Array(CHOLMOD.Dense(HTau[]))),
                    SparseMatrixCSC(min(size(A)...), R_.n, R_.colptr, R_.rowval, R_.nzval),
                    p, hpinv)
end


"""
Solves A x = b using the QR factorization of A and store the results in xr
"""
function solve_qr!(m, n, xr, b, Q, R, Prow, Pcol)
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
  mul!(xr, Q', b[Prow])  # xr ← Q'(P₁b)  NB: using @views here results in tons of allocations?!
  @views x = xr[1:n]
  ldiv!(LinearAlgebra.UpperTriangular(R), x)  # x ← R⁻¹ x
  @views x[Pcol] .= x
  @views r = xr[n+1:m]  # = Q₂'b
  return x, r
end


"""
Computes the QR factorization matrices Qλ and Rλ of [A; √λI]
given the QR factorization of A by performing Givens rotations
If A = QR, we transform [R; 0; √λI] into [Rλ; 0; 0] by performing Givens
rotations that we store in G_list and then Qλ = [ [Q  0]; [0  I] ] * Gᵀ
"""
function fullQR_givens!(R, G_list, news, sqrtλ, nvar, ncon)
	n = nvar
	m = ncon
	counter = 1

	for k = n : -1 : 1
		# We rotate row k of R with row k of √λI to eliminate [k, k]
		G, r = givens(R[k, k], sqrtλ, k, m + k)
		apply_givens!(R, G, r, news, n)
		G_list[counter] = G
		counter += 1
		# print("\n\n", R)

		for l = 1 : n - k
			# We rotate row k + l of R with row k of √λI to eliminate [k, k + l]
			G, r = givens(R[k + l, k + l], news[k + l], k + l, m + k)
			apply_givens!(R, G, r, news, n)
			G_list[counter] = G
			counter += 1
			# print("\n\n", R)
		end
	end
end


"""
Performs the Givens rotation G on [R; 0; √λI] knowing the news
elements in the √λI part and returns the new elements created
"""
function apply_givens!(R, G, r, news, n)
	if G.i1 == n
		R[G.i1, G.i1] = r
		for j = G.i1 + 1 : n
			R[G.i1, j], news[j] = G.c * R[G.i1, j], G.s * R[G.i1, j] # as √λI[G.i2, j] = 0
		end
	else
		R[G.i1, G.i1] = r
		news[G.i1] = 0
		for j = G.i1 + 1 : n
			R[G.i1, j], news[j] = G.c * R[G.i1, j] - G.s * news[j], G.s * R[G.i1, j] + G.c * news[j]
		end
	end
end


"""
Computes Qλᵀ * x where Qλ = [ [Q  0]; [0  I] ] * Gᵀ
Qλᵀ * x = G * [Qᵀx₁; x₂]
"""
function Qλt_mul!(xr, Q, G_list, x, n, m)
	mul!(xr[1:n], Q', x[1:n])
	xr[n + 1 : n + m] = @views x[n + 1 : n + m]
	for k = 1 : size(G_list, 1)
		G = G_list[k]
		xr[G.i1], xr[G.i2] = G.c * xr[G.i1] - G.s * xr[G.i2], G.s * xr[G.i1] + G.c * xr[G.i2]
	end
	return xr
end


# Uncomment to test fullQR_givens

# m = 7
# n = 5
# λ = 1.5
# A = rand([-5.0, 5.0], m, n)
# QR_A = qr(A)
# G_list = Vector{LinearAlgebra.Givens{Float64}}(undef, Int(n*(n + 1)/2))
# news = Vector{Float64}(undef, n)
# fullQR_givens!(QR_A.R, G_list, news, sqrt(λ), n, m)
#
#
# AI = [A; sqrt(λ) * Matrix{Float64}(I, n, n)]
# QR_AI = qr(AI)
#
# print("\n\n", norm(QR_AI.R - QR_A.R[1:n,:]))



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
