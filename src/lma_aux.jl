using LinearAlgebra
using SparseArrays
using BFloat16s


# Add cos, sin, eps, parse and ini_dec functions for BFloat16
import Base: cos, sin, eps, parse, iszero, sign_mask
import Printf: ini_dec

for F in (:cos, :sin)
  @eval begin
    $F(x::BFloat16) = BFloat16($F(Float32(x)))
  end
end

parse(::Type{BFloat16}, x::AbstractString) = BFloat16(parse(Float32, x::AbstractString))

eps(::Type{BFloat16}) = Base.bitcast(BFloat16, 0x3c00)

@eval begin
  $:ini_dec(x::BFloat16, n::Int, digits) = $:ini_dec(Float32(x), n, digits)
end

iszero(x::BFloat16) = reinterpret(UInt16, x) & ~sign_mask(BFloat16) == 0x0000


"""
Normalize the matrix A in the LDL version with BFloat16
"""
function normalize_F162!(A_bis, D, A_norm, n, T)
  D .= T(1)
  one = ones(T, n)
  tol = 1e-4
  first = true

  while first || maximum(abs.(D - one)) <= tol
    for j  = 1 : n
      @views colj = A_bis.nzval[A_bis.colptr[j] : A_bis.colptr[j + 1] - 1]
      D[j] = norm(colj)
      @inbounds A_bis.nzval[A_bis.colptr[j] : A_bis.colptr[j + 1] - 1] /= D[j]
    end
    first = false
  end
  θ = 0.1
  # max_nb = 3.29e38
  max_nb = 6.55e4
  cst = θ * max_nb
  μ = cst
  D /= μ
  # print("\n", maximum(A_bis))
  @views A_norm.nzval .= Float16.(μ * A_bis.nzval)
end


function normalize_F16!(A_bis, D, A_norm, m, n, nz, rows, cols, vals)
  # The m first columns contain A[1,1] and the rows of the jacobian
  # the n next columns contain A[m + 1, m + 1] and the columns of the jacobian
  D[1 : m] .= A_bis[1, 1]^2
  D[m + 1 : m + n] .= A_bis[m + 1, m + 1]^2

  for k = 1 : nz
    i = rows[k]
    j = cols[k]
    D[i] += vals[k]^2
    D[j] += vals[k]^2
  end

  D .= D.^0.25

  for j = 1 : n + m
    for k = A_bis.colptr[j] : A_bis.colptr[j + 1] - 1
      A_bis.nzval[k] /= D[A_bis.rowval[k]] * D[j]
    end
  end

  θ = 0.1
  # max_nb = 3.29e38
  max_nb = 6.55e4
  cst = θ * max_nb
  μ = cst
  D /= μ
  print("\n", maximum(A_bis))
  print("\nok\n", norm(A_bis))
  @views A_norm.nzval .= Float16.(μ * A_bis.nzval)
  print("\nok\n", norm(A_norm))
  print("\n", maximum(A_norm))
end


function normalize_vect!(xr, b, D, n, m)
  for j = 1 : n
    xr[j] = b[j] / D[j]
  end
  xr[n + 1 : n + m] .= 0
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
    if col_norms[j] != 0
      @inbounds A.nzval[A.colptr[j] : A.colptr[j+1] - 1] /= col_norms[j]
    end
  end
end


"""
Denormalize the sparse matrix A
"""
function denormalize_qr!(A, col_norms, n)
  for j = 1 : n
    if col_norms[j] != 0
      A.nzval[A.colptr[j] : A.colptr[j+1] - 1] *= col_norms[j]
    end
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
    if col_norms[j] != 0
      @inbounds A.nzval[A.colptr[m + j] : A.colptr[m + j + 1] - 2] /= col_norms[j]
      # -λ is multiplied by D[j,j]²
      A.nzval[A.colptr[m + j + 1] - 1] /= col_norms[j]^2
    end
  end
end


"""
Denormalize the submatrix J in the matrix A = [ [I J]; [Jᵀ -λI] ]
"""
function denormalize_ldl!(A, col_norms, n, m)
  for j = 1 : n
    if col_norms[j] != 0
      A.nzval[A.colptr[m + j] : A.colptr[m + j + 1] - 2] *= col_norms[j]
    end
  end
end


"""
Denormalize (in place) the vector x
"""
function denormalize_vect!(x, col_norms, n)
  for j = 1 : n
    if col_norms[j] != 0
      x[j] /= col_norms[j]
    end
  end
end


"""
Returns true if the jacobian is not nearly constant
ie: ‖Jₖ₋₁ - Jₖ‖ > tol
"""
function jac_not_const(Jδ, Jδ_suiv, rows, cols, vals, δ, n, tol)
  mul_sparse!(Jδ_suiv, rows, cols, vals, δ, n)
  return norm(Jδ - Jδ_suiv) > tol * norm(Jδ)
end


"""
Multiply (in place) a sparse matrix (with n non-zeros) with a vector
"""
function mul_sparse!(xr, rows, cols, vals, x, n)
  xr .= 0
  for k = 1 : n
    xr[rows[k]] += vals[k] * x[cols[k]]
  end
  return xr
end


"""
Multiply a sparse matrix (with n non-zeros) with a vector
"""
function mul_sparse(rows, cols, vals, x, n, l)
  xr = zeros(eltype(x), l)
  for k = 1 : n
    xr[rows[k]] += vals[k] * x[cols[k]]
  end
  return xr
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
