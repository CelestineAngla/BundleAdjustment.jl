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
  ldiv!(LinearAlgebra.UpperTriangular(R), x)  # x ← R⁻¹ x
  @views x[Pcol] .= x
  @views r = xr[n+1:m]  # = Q₂'b
  return x, r
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
Computes the QR factorization matrices Qλ and Rλ of [A; √λI]
given the QR factorization of A by performing Givens rotations
If A = QR, we transform [R; 0; √λI] into [Rλ; 0; 0] by performing Givens
rotations that we store in G_list and then Qλ = [ [Q  0]; [0  I] ] * Gᵀ
"""
function fullQR_givens!(R, Rt, G_list, news, sqrtλ, n, m, nnz_R)

	counter = 1
	# print("\nbegin\n\n", R)
	for k = n : -1 : 1
	    # We rotate row k of R with row k of √λI to eliminate [k, k]
	    G, r = givens(R[k, k], sqrtλ, k, m + k)
		min_news = apply_givens!(R, Rt, G, r, news, n, m, true, 0)
		G_list[counter] = G
		counter += 1
		# print("\n\n", news)
		# print("\n\n", R)

		if Rt.colptr[k] < nnz_R
		  beg = Rt.rowval[Rt.colptr[k] + 1]
	    else
		  beg = n + 1
	    end
		for col = beg : n
	      if news[col] != 0
	        # We rotate row l of R with row k of √λI to eliminate [k, l]
	  	    G, r = givens(R[col, col], news[col], col, m + k)
			min_news = apply_givens!(R, Rt, G, r, news, n, m, false, min_news + 1)
			G_list[counter] = G
	  		counter += 1
			# print("\n\n", news)
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
function apply_givens!(R, Rt, G, r, news, n, m, diag, old_min_news)
	# If we want to eliminate the diagonal element (ie: √λ),
	# we know that news is empty so far
	min_news = 0
	min_found = false
	if diag
		# print("\n i :", G.i1, " j ", G.i1, " r ", r)
		R[G.i1, G.i1] = r
		for k = Rt.colptr[G.i1] + 1 : Rt.colptr[G.i1 + 1] - 1
			col = Rt.rowval[k]
			# print("\n i :", G.i1, " j ", col)
        	R[G.i1, col], news[col] = G.c * R[G.i1, col], - G.s * R[G.i1, col]
			if !min_found && news[col] != 0
			  min_found = true
			  min_news = col
		    end
    	end

	# Otherwise we eliminate the first non-zero element of news
	else
		# print("\n i :", G.i1, " j ", G.i1, " r ", r)
		R[G.i1, G.i1] = r
		news[G.i1] = 0
		# print("\n else", old_min_news)
		if old_min_news != 0
		  beg = old_min_news
	    else
		  beg = n + 1
	    end
		# print("\n beg ", beg)
		for col = beg : n
			# print("\n i :", G.i1, " j ", col)
      		R[G.i1, col], news[col] = G.c * R[G.i1, col] + G.s * news[col], - G.s * R[G.i1, col] + G.c * news[col]
			if !min_found && news[col] != 0
			  min_found = true
			  min_news = col
		    end
    	end
	end
	return min_news
end



"""
Computes Qλᵀ * x where Qλ = [ [Q  0]; [0  I] ] * Gᵀ
Qλᵀ * x = G * [Qᵀx₁; x₂]
"""
function Qλt_mul!(xr, Q, G_list, x, n, m, counter)
	@views mul!(xr[1:m], Q', x[1:m])
	xr[m + 1 : m + n] = @views x[m + 1 : m + n]
	for k = 1 : counter
		G = G_list[k]
		xr[G.i1], xr[G.i2] = G.c * xr[G.i1] + G.s * xr[G.i2], - G.s * xr[G.i1] + G.c * xr[G.i2]
	end
	return xr
end

function Qλt_mul_verif!(xr, Q, G_list, x, n, m, counter)
	QI = zeros(n+m, n+m)
	QI[1:m, 1:m] .= Q
	QI[m+1:m+n, m+1:m+n] .= Matrix{Float64}(I, n, n)
	Qλ = similar(QI)
	for k = 1 : counter
		Qλ = QI * G_list[k]'
		QI .= Qλ
	end
	mul!(xr, Qλ', x)
	return Qλ
end


# Uncomment to test fullQR_givens
#
# m = 7
# n = 5
# λ = 1.5
# rows = rand(1:m, 8)
# cols = rand(1:n, 8)
# vals = rand(-4.5:4.5, 8)
# A = sparse(rows, cols, vals, m, n)
# print("\n\nA\n\n", A)
#
# b = rand(-4.5:4.5, m+n)
#
#
#
# # QR facto of A
# QR_A = qr(A)
# print("\n\nPcol & Prow for A :\n", QR_A.pcol, "\n", QR_A.prow)
# print("\n\n", QR_A.Q * vcat(QR_A.R, zeros(m-n, n)))
#
# # A_R is [R; √λI]
# A_R = zeros(m+n, n)
# A_R[1:n, 1:n] .= QR_A.R
# for k = 1:n
# 	A_R[m+k, k] = sqrt(λ)
# end
#
# # Givens rotations on QR_A.R
# G_list = Vector{LinearAlgebra.Givens{Float64}}(undef, Int(n*(n + 1)/2))
# news = zeros(Float64, n)
# # news = sparsevec(I,V)
# Rt = sparse(QR_A.R')
# col_norms = ones(n)
# counter = fullQR_givens!(QR_A.R, Rt, G_list, news, sqrt(λ), col_norms, n, m, nnz(QR_A.R))
#
# # performs the same Givens rotations on A_R
# A_R2 = similar(A_R)
# for k = 1 : counter
# 	A_R2 = G_list[k] * A_R
# 	A_R .= A_R2
# end
# print("\n\n", A_R)
#
# # Check if the λ have been eliminated in A_R and if A_R = Rλ
# print("\n\n", norm(A_R[n+1:n+m, :]), "\n", norm(A_R[1:n, :] - QR_A.R))
#
#
# # Check if Qλt_mul! works well
# my_xr = similar(b)
# Qλt_mul!(my_xr, QR_A.Q, G_list, b, n, m, counter)
# true_xr = similar(b)
# Qλ = Qλt_mul_verif!(true_xr, QR_A.Q, G_list, b, n, m, counter)
# print("\n\nQλt\n\n", norm(my_xr - true_xr))

# # Solve [A; √λ] x = b
# xr1 = similar(b)
# Prow = vcat(QR_A.prow, collect(m + 1 : m + n))
# δ1, δr1 = solve_qr2!(m+n, n, xr1, b, QR_A.Q, QR_A.R, Prow, QR_A.pcol, counter, G_list)
#
#
# # Solve [A; √λ] x = b with QR factorization of [A; √λ]
# AI = [A; sqrt(λ) * Matrix{Float64}(I, n, n)]
# QR_AI = qr(AI)
# print("\n\nPcol & Prow for AI :\n", QR_AI.pcol, "\n", QR_AI.prow)
# print("\n\n AI\n", AI)
# print("\n\n QR\n", QR_AI.Q * vcat(QR_AI.R, zeros(m, n)))
# print("\n\nPcol & Prow for A :\n", QR_A.pcol, "\n", QR_A.prow)
# print("\n\n", Qλ * vcat(QR_A.R, zeros(m, n)))
#
# xr2 = similar(b)
# δ2, δr2 = solve_qr!(m+n, n, xr2, b, QR_AI.Q, QR_AI.R, QR_AI.prow, QR_AI.pcol)
#
# # Check the results
# print("\n\n Results : ")
# print("\n\n", δ1, "\n\n", δ2, "\n\n", norm(δ1 - δ2))
# print("\n\nb\n", b)
# print("\n\n", AI * δ1, "\n\n", AI * δ2)
# print("\n\n", AI * δ1 - b, "\n\n", AI * δ2 - b)
