using LinearAlgebra
using SparseArrays
using .Threads

using DataFrames
using JLD
function save_stats(stats::Dict{Symbol,DataFrame}, filename::AbstractString; force::Bool=false, key::String="stats")
  isfile(filename) && !force && error("$filename already exists; use `force=true` to overwrite")
  jldopen(filename, "w") do file
    file[key] = stats
  end
end


"""
Normalize (in place) the n columns of a sparse matrix A
and return the vector of norms
"""
function normalize_qr_a!(A, col_norms, n)
  for j = 1 : n
      @views colj = A.nzval[A.colptr[j] : A.colptr[j+1] - 1]
      col_norms[j] = norm(colj)
      @inbounds A.nzval[A.colptr[j] : A.colptr[j+1] - 1] /= col_norms[j]
  end
end


"""
Normalize the submatrix J in the matrix A = [J; √λI]
and multiply the √λI part by D
"""
function normalize_qr_j!(A, col_norms, n)
  for j = 1 : n
    # the √λ element does not count in the norm of J
    @views colj = A.nzval[A.colptr[j] : A.colptr[j+1] - 2]
    col_norms[j] = norm(colj)
    @inbounds A.nzval[A.colptr[j] : A.colptr[j+1] - 1] /= col_norms[j]
  end
end


"""
Denormalize the sparse matrix A
"""
function denormalize_qr!(A, col_norms, n)
  for j = 1 : n
    A.nzval[A.colptr[j] : A.colptr[j+1] - 1] *= col_norms[j]
  end
end


"""
Normalize the submatrix J in the matrix A = [ [I J]; [Jᵀ -λI] ]
and multiply the -λI part by D²
"""
function normalize_ldl!(A, col_norms, n, m)
  for j = 1 : n
    # the -λ element does not count in the norm of J
    @views colj = A.nzval[A.colptr[m + j] : A.colptr[m + j + 1] - 2]
    col_norms[j] = norm(colj)
    @inbounds A.nzval[A.colptr[m + j] : A.colptr[m + j + 1] - 2] /= col_norms[j]
    # -λ is multiplied by D[j,j]²
    A.nzval[A.colptr[m + j + 1] - 1] /= col_norms[j]^2
  end
end


"""
Denormalize the submatrix J in the matrix A = [ [I J]; [Jᵀ -λI] ]
"""
function denormalize_ldl!(A, col_norms, n, m)
  for j = 1 : n
    A.nzval[A.colptr[m + j] : A.colptr[m + j + 1] - 2] *= col_norms[j]
  end
end


"""
Denormalize (in place) the vector x
"""
function denormalize_vect!(x, col_norms, n)
  for j = 1 : n
    x[j] /= col_norms[j]
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
function jac_not_const(Jδ, Jδ_suiv, rows, cols, vals, δ, n, tol)
  mul_sparse!(Jδ_suiv, rows, cols, vals, δ, n)
  return norm(Jδ - Jδ_suiv) > tol * norm(Jδ)
end


function mul_sparse!(xr, rows, cols, vals, x, n)
  xr .= 0
  for k = 1 : n
    xr[rows[k]] += vals[k] * x[cols[k]]
  end
  return xr
end

function mul_sparse(rows, cols, vals, x, n, l)
  xr = zeros(eltype(x), l)
  for k = 1 : n
    xr[rows[k]] += vals[k] * x[cols[k]]
  end
  return xr
end


# m = 7
# n = 5
# nz = 5
# rows = rand(1:m, nz)
# cols = rand(1:n, nz)
# vals = rand(-4.5:4.5, nz)
# A = sparse(rows, cols, vals, m, n)
#
# x = rand(-4.5:4.5, n)
# xr = Vector{Float64}(undef, m)
#
# @time begin
# mul!(xr, A, x)
# end
# print(xr)
#
# @time begin
# mul_sparse!(xr, rows, cols, vals, x, nz)
# end
# print(xr)


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
