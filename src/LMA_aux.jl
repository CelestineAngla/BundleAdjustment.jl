using LDLFactorizations
using LinearAlgebra

"""
Solves A x = b using the LDL factorization of A (A is symetric)
"""
function ldl_solve1!(n, b, Lp, Li, Lx, D, P)
  @views y = b[P]
  ldl_lsolve!(n, y, Lp, Li, Lx)
  ldl_dsolve!(n, y, D)
  ldl_ltsolve!(n, y, Lp, Li, Lx)
  @views b[P] = y
  return b
end

function ldl_solve2!(n, y, Lp, Li, Lx, D, P)
	permute!(y, P)
  ldl_lsolve!(n, y, Lp, Li, Lx)
  ldl_dsolve!(n, y, D)
  ldl_ltsolve!(n, y, Lp, Li, Lx)
	invpermute!(y, P)
  return y
end

function ldl_solve3!(n, b, Lp, Li, Lx, D, P)
  y = b[P]
  ldl_lsolve!(n, y, Lp, Li, Lx)
  ldl_dsolve!(n, y, D)
  ldl_ltsolve!(n, y, Lp, Li, Lx)
  b[P] = y
  return b
end

function ldl_solve4!(n, y, Lp, Li, Lx, D, P)
  permutation!(y, P, n)
  ldl_lsolve!(n, y, Lp, Li, Lx)
  ldl_dsolve!(n, y, D)
  ldl_ltsolve!(n, y, Lp, Li, Lx)
  permutation_inv!(y, P, n)
  return y
end

"""
x = x[P]
"""
function permutation!(x, P, n)
	i = 1
	while i < n
		current = i
		suiv_i = i + 1
		print("\n\ni: ", i, " curr: ", current, " suiv_i: ", suiv_i, " P[curr]: ", P[current])
		while i != P[current]
			next = P[current]
			if next == suiv_i
				suiv_i += 1
			end
			x[current], x[next] = x[next], x[current]
			current = next
			print("\nnext: ", next, " suiv_i: ", suiv_i, " current: ", current, " P[curr]: ", P[current], " x: ", x)
		end
		i = suiv_i
	end
end



"""
x[P] = x
"""
function permutation_inv!(x, P, n)
	k = 1
	i = 1
	p = P[i]
	tmp = x[p]
	x[p] = x[i]
	while k != n
		i = P[i]
		p = P[i]
		tmp, x[p] = x[p], tmp
		k += 1
	end
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

# Uncomment to compare ldl_solve1 and ldl_solve2

# using BenchmarkTools
# A = sparse([1, 1, 1, 2, 2, 3], [1, 2, 3, 2, 3, 3], [1.0, 4.0, -3.0, 1.0, -2.0, 2.0])
# b = [1.0, -4.0, 0.0]
# X = similar(b)
# LDLT = ldl(A, upper=true)
# @btime begin
# X .= b
# ldl_solve1!(3, X, LDLT.L.colptr, LDLT.L.rowval, LDLT.L.nzval, LDLT.D, LDLT.P)
# end
# print(X)
# print(b, "\n")
# @btime begin
# X .= b
# ldl_solve2!(3, X, LDLT.L.colptr, LDLT.L.rowval, LDLT.L.nzval, LDLT.D, LDLT.P)
# end
# print(X)
# print(b)


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

# using SparseArrays
# rows = rand(1:4, 5)
# cols = rand(1:4, 5)
# vals = rand(-3.5:3.5, 5)
# A = sparse(rows, cols, vals, 4, 4)
# print(A)
# QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_METIS)

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
function fullQR_Givens!(Q, R, sqrtλ, ncon, nvar)
	n = nvar
	m = ncon
	G_list = Vector{Givens{Float64}}(undef, n*(n + 1)/2)
	counter = 1

	for k = 1 : n
		l = n - k + 1
		G, _ = givens(Rλ, l, m + l, l)
		G_list[counter] = G
		counter += 1
		new = apply_givens!(R, sqrtλ, G, new)
		# print("\n\n", Rλ)

		for i = 1 : k - 1
			G, _ = givens(Rλ, n - k + i + 1, m + n - k + 1, n - k + i + 1)
			G_list[counter] = G
			counter += 1
			new = apply_givens!(R, sqrtλ, G, new)
			# print("\n\n", Rλ)
		end
	end
	return Qλ, Rλ
end


"""
Performs the Givens rotation G on [R; 0; √λI] knowing the news
elements in the √λI part and returns the new elements created
"""
function apply_givens!(R, sqrtλ, G, new)
	# new list of line i in √λI
	# if i,i (ie remove √λ) new = ...
	# if i,j (j > i, remove new) new[j] = 0 update other new
	# consider only nnz elements
	return new
end

"""
Computes X = GX where G is a Givens rotation
"""
function mul_givens!(G, X)
	for k = 1 : size(X, 2)
		X[G.i1, k], X[G.i2, k] = G.c * X[G.i1, k] + G.s * X[G.i2, k], -G.s * X[G.i1, k] + G.c * X[G.i2, k]
	end
	return X
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


# Uncomment to test fullQR_Givens

# m = 7
# n = 5
# λ = 1.5
# A = rand([-5.0, 5.0], m, n)
# @time begin
# QR_A = qr(A)
# Qλ, Rλ = fullQR_Givens!(QR_A.Q, QR_A.R, λ, m, n)
# end
#
# AI = [A; sqrt(λ) * Matrix{Float64}(I, n, n)]
# QR_AI = qr(AI)
#
# print("\n\n", norm(Rλ[6:12, :]), "\n\n", norm(QR_AI.Q - Qλ), "\n\n", norm(QR_AI.R - Rλ[1:n,:]))



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
