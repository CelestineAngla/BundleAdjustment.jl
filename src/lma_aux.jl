using LinearAlgebra
using SparseArrays
using .Threads


# """
# Solves the linear problem Ax = b with QR or LDL factorization
# """
# function solve_linear(facto, perm, P, A, b, xr)
#   if facto == :QR
# 	 # Solve min ||[J √λI] δ + [r 0]||² with QR factorization
# 		if perm == :AMD
# 			QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_AMD)
# 		elseif perm == :Metis
# 			QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_METIS)
# 		end
# 		δ, δr = solve_qr!(model.nls_meta.nequ + model.meta.nvar, model.meta.nvar, xr, b, QR.Q, QR.R, QR.prow, QR.pcol)
#
# 	elseif facto == :LDL
# 		# Solve [[I J]; [Jᵀ - λI]] X = [r; 0] with LDL factorization
# 		LDLT = ldl(A, P, upper=true)
# 		LDLT = ldl_numeric_aux(A, P, model.meta.nvar + model.nls_meta.nequ, Cp, Ci, Lp, parent, Lnz, Li, Lx, D, Y, pattern, flag, pinv)
# 		xr .= b
# 		ldl_solve!(model.nls_meta.nequ + model.meta.nvar, xr, LDLT.L.colptr, LDLT.L.rowval, LDLT.L.nzval, LDLT.D, P)
# 		δr = xr[1 : model.nls_meta.nequ]
# 		δ = xr[model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar]
# 	end
# end


"""
Normalize (in place) the n columns of a sparse matrix A
and return the vector of norms
"""
function normalize_cols!(A, col_norms, n)
  q, re = divrem(n, nthreads())
  if re != 0
    q += 1
  end

  @threads for t = 1 : nthreads()
    @simd for j = 1 + (t - 1) * q : min(t * q, n)
      @views colj = A.nzval[A.colptr[j] : A.colptr[j+1] - 1]
      col_norms[j] = norm(colj)
      @inbounds A.nzval[A.colptr[j] : A.colptr[j+1] - 1] /= col_norms[j]
    end
  end
end

"""
Denormalize (in place) the sparse matrix A
"""
function denormalize_cols!(A, col_norms, n)
  q, re = divrem(n, nthreads())
  if re != 0
    q += 1
  end

  @threads for t = 1 : nthreads()
    @simd for j = 1 + (t - 1) * q : min(t * q, n)
      A.nzval[A.colptr[j] : A.colptr[j+1] - 1] *= col_norms[j]
    end
  end
end

"""
Denormalize (in place) the vector x
"""
function denormalize!(x, col_norms, n)
  q, re = divrem(n, nthreads())
  if re != 0
    q += 1
  end

  @threads for t = 1 : nthreads()
    @simd for j = 1 + (t - 1) * q : min(t * q, n)
      x[j] /= col_norms[j]
    end
  end
end

# Uncomment to test normalize_cols! and denormalize_cols!

# m = 7
# n = 5
# rows = rand(1:m, 5)
# cols = rand(1:n, 5)
# vals = rand(-4.5:4.5, 5)
# A = sparse(rows, cols, vals, m, n)
# col_norms = Vector{eltype(A)}(undef, n)
# print(A)
# normalize_cols!(A, col_norms, n)
# print("\n\n", A)
# print("\n\n", col_norms)
# denormalize_cols!(A, col_norms, n)
# print("\n\n", A)


"""
Returns true if the jacobian is not nearly constant
ie: ‖Jₖ₋₁ - Jₖ‖ > tol
"""
function jac_not_const(old_vals, new_vals, tol)
  return norm(old_vals - new_vals) > tol
end

"""
Fills the vector b with the p last ‖r(xⱼ)‖
"""
function fill_b_reglin!(b, p, iter, sq_norm_r)
  if iter <= p - 1
    b[iter + 1] = sqrt(sq_norm_r)
  else
    b[1 : p - 1] = b[2 : p]
    b[p] = sqrt(sq_norm_r)
  end
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
