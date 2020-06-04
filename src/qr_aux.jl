using LinearAlgebra
using SparseArrays


using SparseArrays: SparseMatrixCSC
import SuiteSparse.SPQR: CHOLMOD, _default_tol, _qr!, QRSparse
import SuiteSparse.CHOLMOD: Sparse
using SuiteSparse

"""
Modified version of the wrapper of SPQR so that the user can choose the ordering
"""
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
Solves min ‖Ax - b‖ using the QR factorization of A and store the results in xr
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
function fullQR_givens!(R, Rt, G_list, news, sqrtλ, n, m, nnz_R)
  counter = 0
  # print("\nbegin\n\n", R)
  for k = n : -1 : 1
    # We rotate row k of R with row k of √λI to eliminate [k, k]
    G, r = givens(R[k, k], sqrtλ, k, m + k)
    min_news = apply_givens!(R, Rt, G, r, news, n, m, true, 0)
    counter += 1
    G_list[counter] = G
    # print("\n\n", news, "\n\n", R)

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
        counter += 1
        G_list[counter] = G
        # print("\n\n", news, "\n\n", R)
      end
    end
  end

  return counter
end


"""
Performs the Givens rotation G on [R; 0; √λI] knowing the news
elements in the √λI part and returns the new elements created
"""
function apply_givens!(R, Rt, G, r, news, n, m, diag, old_min_news)
  min_news = 0
  min_found = false

  # If we want to eliminate the diagonal element (ie: √λ),
  # we know that news is empty so far
  if diag
    R[G.i1, G.i1] = r
    for k = Rt.colptr[G.i1] + 1 : Rt.colptr[G.i1 + 1] - 1
      col = Rt.rowval[k]
      R[G.i1, col], news[col] = G.c * R[G.i1, col], - G.s * R[G.i1, col]
      if !min_found && news[col] != 0
        min_found = true
        min_news = col
      end
    end

  # Otherwise we eliminate the first non-zero element of news
  else
    R[G.i1, G.i1] = r
    news[G.i1] = 0
    if old_min_news != 0
      beg = old_min_news
    else
      beg = n + 1
    end
    for col = beg : n
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
Solves ‖Ax - b‖ using the QR factorization (with Givens strategy) of A and store the results in xr
"""
function solve_qr2!(m, n, xr, b, Q, R, Prow, Pcol, counter, G_list)
  m ≥ n || error("currently, this function only supports overdetermined problems")
  @assert length(b) == m
  @assert length(xr) == m
  Qλt_mul!(xr, Q, G_list, b[Prow], n, m - n, counter)
  @views x = xr[1:n]
  ldiv!(LinearAlgebra.UpperTriangular(R), x)  # x ← R⁻¹ x
  @views x[Pcol] .= x
  @views r = xr[n+1:m]  # = Q₂'b
  return x, r
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

"""
Function to test Qλt_mul!
"""
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
