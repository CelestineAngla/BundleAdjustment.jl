using LinearAlgebra
using SparseArrays


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

function solve_qr2!(m, n, xr, b, Q, R, Prow, Pcol, counter, G_list)
  m ≥ n || error("currently, this function only supports overdetermined problems")
  @assert length(b) == m
  @assert length(xr) == m

  Qλt_mul!(xr, Q, G_list, b[Prow], n, m-n, counter)
	@views x = xr[1:n]
  print("\n", size(R), " ", size(x))
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
	# print("\n\n", R)

	for k = n : -1 : 1
		# print("\n k : ", k)

    # We find the first non-zero element of column k
    i = k
    while R[i, k] == 0
      i -= 1
    end
    # print("\n i : ", i)

    # We rotate row i of R with row k of √λI to eliminate [k, k]
    G, r = givens(R[i, k], sqrtλ, i, n + k)
    apply_givens!(R, G, r, news, n, true)
		# print("\n news \n", news)
		G_list[counter] = G
		counter += 1
		# print("\n\n", R)

    for l = i : n
      # print("\n l : ", l)
      if news[l] != 0

        # We find the first non-zero element of column l
        j = l
        while R[j, l] == 0
          j -= 1
        end
        # print("\n j : ", j)

        # We rotate row j of R with row k of √λI to eliminate [k, l]
  			G, r = givens(R[j, l], news[l], j, n + l)
        apply_givens!(R, G, r, news, n, false)
  			# print("\n news \n", news)
  			G_list[counter] = G
  			counter += 1
  			# print("\n\n", R)

      end
    end

	end
  return counter - 1
end


"""
Performs the Givens rotation G on [R; 0; √λI] knowing the news
elements in the √λI part and returns the new elements created
"""
function apply_givens!(R, G, r, news, n, diag)
	# print("\n i1 : ", G.i1, " i2 : ", G.i2, " c : ", G.c, " s : ", G.s)
	# If we want to eliminate the diagonal element (ie: √λ),
	# we know that news is empty so far
	if diag
    for j = G.i1 : n
      if j == G.i2 - n
        R[G.i1, G.i2 - n] = r
      else
        R[G.i1, j], news[j] = G.c * R[G.i1, j], - G.s * R[G.i1, j]
      end
    end

	# Otherwise we eliminate the first non-zero element of news
	else
    for j = G.i1 : n
      if j == G.i2 - n
        R[G.i1, G.i2 - n] = r
        news[G.i2 - n] = 0
      else
			  R[G.i1, j], news[j] = G.c * R[G.i1, j] + G.s * news[j], - G.s * R[G.i1, j] + G.c * news[j]
      end
    end
	end
end


"""
Computes Qλᵀ * x where Qλ = [ [Q  0]; [0  I] ] * Gᵀ
Qλᵀ * x = G * [Qᵀx₁; x₂]
"""
function Qλt_mul!(xr, Q, G_list, x, n, m, counter)
	mul!(xr[1:m], Q', x[1:m])
	xr[m + 1 : m + n] = @views x[m + 1 : m + n]
	for k = 1 : counter
		G = G_list[k]
		xr[G.i1], xr[G.i2] = G.c * xr[G.i1] - G.s * xr[G.i2], G.s * xr[G.i1] + G.c * xr[G.i2]
	end
	return xr
end


# Uncomment to test fullQR_givens

# m = 7
# n = 5
# λ = 1.5
# rows = rand(1:m, 5)
# cols = rand(1:n, 5)
# vals = rand(-4.5:4.5, 5)
# A = sparse(rows, cols, vals, m, n)
# for j = 1 : n
#   if norm(A[:,j]) == 0
#     i = rand(1 : m)
#     A[i, j] = rand(-4.5:4.5)
#   end
# end
# b = rand(-4.5:4.5, m+n)
# QR_A = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_NATURAL)
# G_list = Vector{LinearAlgebra.Givens{Float64}}(undef, Int(n*(n + 1)/2))
# news = Vector{Float64}(undef, n)
# counter = fullQR_givens!(QR_A.R, G_list, news, sqrt(λ), n, m)
#
#
# AI = [A; sqrt(λ) * Matrix{Float64}(I, n, n)]
# QR_AI = myqr(AI, ordering=SuiteSparse.SPQR.ORDERING_NATURAL)
# xr = similar(b)
# Qλt_mul!(xr, QR_A.Q, G_list, b, n, m, counter)
#
# print("\n\n", QR_A.R, "\n\n", QR_AI.R)
#
# inv_A_R = inv(Matrix(QR_A.R))
# inv_AI_R = inv(Matrix(QR_AI.R))
#
# inv_A_R = hcat(inv_A_R, zeros(n, m))
# inv_AI_R = hcat(inv_AI_R, zeros(n, m))
#
# print("\n", size(inv_A_R), " ", size(inv_AI_R), " ", size(QR_AI.Q'))
#
# print("\n\n", norm(QR_AI.R - QR_A.R), "\n", norm(inv_A_R * xr - inv_AI_R * QR_AI.Q' * b))
