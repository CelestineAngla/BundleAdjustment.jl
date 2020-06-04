using LDLFactorizations


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


"""
Solves A x = b using the LDL factorization of A (A is symetric)
"""
function ldl_solve!(n, b, Lp, Li, Lx, D, P)
  @views y = b[P]
  ldl_lsolve!(n, y, Lp, Li, Lx)
  ldl_dsolve!(n, y, D)
  ldl_ltsolve!(n, y, Lp, Li, Lx)
  return b
end


function col_symb!(n, Ap, Ai, Cp, w, Pinv)
  fill!(w, 0)
  @inbounds for j = 1:n
    @inbounds for p = Ap[j] : (Ap[j+1]-1)
      i = Ai[p]
      i >= j && break  # only upper part
      Pinv[i] < Pinv[j] && continue  # store only what will be used during the factorization
      w[i] += 1  # count entries
    end
  end

  Cp[1] = 1
  @inbounds for i = 1:n  # cumulative sum
    Cp[i+1] = w[i] + Cp[i]
    w[i] = Cp[i]
  end
end


function col_num!(n, Ap, Ai, Ax, Cp, Ci, w, Pinv)
  @inbounds for j = 1:n
    @inbounds for p = Ap[j] : (Ap[j+1]-1)
      i = Ai[p]
      i >= j  && break  # only upper part
      Pinv[i] < Pinv[j] && continue  # store only what will be used during the factorization
      Ci[w[i]] = j
      w[i] += 1
    end
  end
end


function ldl_symbolic_upper!(n, Ap, Ai, Cp, Ci , Lp, parent, Lnz, flag, P, Pinv)
  @inbounds for k = 1:n
    parent[k] = -1
    flag[k] = k
    Lnz[k] = 0
    pk = P[k]
    @inbounds for p = Ap[pk] : (Ap[pk+1] - 1)
      i = Pinv[Ai[p]]
      i ≥ k && continue
      @inbounds while flag[i] != k
        if parent[i] == -1
          parent[i] = k
        end
        Lnz[i] += 1
        flag[i] = k
        i = parent[i]
      end
    end

    # Missing nonzero elements of the upper triangle
    @inbounds for ind = Cp[pk] : (Cp[pk+1] - 1)
      i = Pinv[Ci[ind]]
      i > k && continue
      @inbounds while flag[i] != k
        if parent[i] == -1
          parent[i] = k
        end
      Lnz[i] += 1
      flag[i] = k
      i = parent[i]
      end
    end
  end
  Lp[1] = 1
  @inbounds for k = 1:n
    Lp[k+1] = Lp[k] + Lnz[k]
  end
end


function ldl_numeric_upper!(n, Ap, Ai, Ax, Cp, Ci, Lp, parent, Lnz, Li, Lx, D, Y,
                      pattern, flag, P, Pinv)
  @inbounds for k = 1:n
    Y[k] = 0
    top = n+1
    flag[k] = k
    Lnz[k] = 0
    pk = P[k]
    @inbounds for p = Ap[pk] : (Ap[pk+1] - 1)
      i = Pinv[Ai[p]]
      i > k && continue
      Y[i] += Ax[p]
      len = 1
      @inbounds while flag[i] != k
        pattern[len] = i
        len += 1
        flag[i] = k
        i = parent[i]
      end
      @inbounds while len > 1
        top -= 1
        len -= 1
        pattern[top] = pattern[len]
      end
    end
    # missing non zero elements of the upper triangle
    @inbounds for ind = Cp[pk] : (Cp[pk+1] - 1)
      i2 = Ci[ind]
      i = Pinv[i2]
      i > k && continue
      @inbounds for p = Ap[i2] : (Ap[i2+1] - 1)
        Ai[p] < pk && continue
        Y[i] += Ax[p]
        len = 1
        @inbounds while flag[i] != k
          pattern[len] = i
          len += 1
          flag[i] = k
          i = parent[i]
        end
        @inbounds while len > 1
          top -= 1
          len -= 1
          pattern[top] = pattern[len]
        end
        break
      end
    end
    D[k] = Y[k]
    Y[k] = 0
    @inbounds while top ≤ n
      i = pattern[top]
      yi = Y[i]
      Y[i] = 0
      @inbounds for p = Lp[i] : (Lp[i] + Lnz[i] - 1)
        Y[Li[p]] -= Lx[p] * yi
      end
      p = Lp[i] + Lnz[i]
      l_ki = yi / D[i]
      D[k] -= l_ki * yi
      Li[p] = k
      Lx[p] = l_ki
      Lnz[i] += 1
      top += 1
    end
    D[k] == 0 && throw(SQDException("matrix does not possess a LDL' factorization for this permutation"))
  end
end


mutable struct LDLFactorization{T<:Real,Ti<:Integer, Ti2<:Integer}
  L::SparseMatrixCSC{T,Ti}
  D::Vector{T}
  P::Vector{Ti2}
end


abstract type AbstractLDLSymbolic end

mutable struct LDLSymbolicUpper{T<:Real,Ti<:Integer, Ti2<:Integer} <: AbstractLDLSymbolic
  n::Ti
  Cp::Vector{Ti}
  Ci::Vector{Ti}
  Lp::Vector{Ti}
  parent::Vector{Ti}
  Lnz::Vector{Ti}
  Li::Vector{Ti}
  Lx::Vector{T}
  D::Vector{T}
  Y::Vector{T}
  pattern::Vector{Ti}
  flag::Vector{Ti}
  P::Vector{Ti2}
  pinv::Vector{Ti}
end

mutable struct LDLSymbolic{T<:Real,Ti<:Integer} <: AbstractLDLSymbolic
  n::Ti
  Lp::Vector{Ti}
  parent::Vector{Ti}
  Lnz::Vector{Ti}
  Li::Vector{Ti}
  Lx::Vector{T}
  D::Vector{T}
  Y::Vector{T}
  pattern::Vector{Ti}
  flag::Vector{Ti}
  P::Vector{Ti}
  pinv::Vector{Ti}
end


function ldl_analyse(A::SparseMatrixCSC{T,Ti}, P::Vector{Ti2}; upper=false, n::Int=size(A,1)) where {T<:Real,Ti<:Integer, Ti2<:Integer}
	# allocate space for symbolic analysis
  parent = Vector{Ti}(undef, n)
  Lnz = Vector{Ti}(undef, n)
  flag = Vector{Ti}(undef, n)
  pinv = Vector{Ti}(undef, n)
  Lp = Vector{Ti}(undef, n+1)

  # Compute inverse permutation
  @inbounds for k = 1:n
    pinv[P[k]] = k
  end

  # perform symbolic analysis
  if upper
  	Cp = Vector{Ti}(undef, n + 1)
  	col_symb!(n, A.colptr, A.rowval, Cp, Lp, pinv)
  	Ci = Vector{Ti}(undef, Cp[end] - 1)
  	col_num!(n, A.colptr, A.rowval, A.nzval, Cp, Ci, Lp, pinv)
  	ldl_symbolic_upper!(n, A.colptr, A.rowval, Cp, Ci, Lp, parent, Lnz, flag, P, pinv)
  else
    ldl_symbolic!(n, A.colptr, A.rowval, Lp, parent, Lnz, flag, P, pinv)
  end

  # allocate space for numerical factorization
	Li = Vector{Ti}(undef, Lp[n] - 1)
	Lx = Vector{T}(undef, Lp[n] - 1)
	Y = Vector{T}(undef, n)
	D = Vector{T}(undef, n)
	pattern = Vector{Ti}(undef, n)

  if upper
    return LDLSymbolicUpper(n, Cp, Ci, Lp, parent, Lnz, Li, Lx, D, Y, pattern, flag, P, pinv)
  else
    return LDLSymbolic(n, Lp, parent, Lnz, Li, Lx, D, Y, pattern, flag, P, pinv)
  end
end


function ldl_factorize(A::SparseMatrixCSC{T,Ti}, LDLSymbolic::AbstractLDLSymbolic, upper=false) where {T<:Real,Ti<:Integer}
  if upper
    ldl_numeric_upper!(LDLSymbolic.n, A.colptr, A.rowval, A.nzval, LDLSymbolic.Cp, LDLSymbolic.Ci, LDLSymbolic.Lp, LDLSymbolic.parent, LDLSymbolic.Lnz, LDLSymbolic.Li, LDLSymbolic.Lx, LDLSymbolic.D, LDLSymbolic.Y, LDLSymbolic.pattern, LDLSymbolic.flag, LDLSymbolic.P, LDLSymbolic.pinv)
  else
    ldl_numeric!(LDLSymbolic.n, A.colptr, A.rowval, A.nzval, LDLSymbolic.Lp, LDLSymbolic.parent, LDLSymbolic.Lnz, LDLSymbolic.Li, LDLSymbolic.Lx, LDLSymbolic.D, LDLSymbolic.Y, LDLSymbolic.pattern, LDLSymbolic.flag, LDLSymbolic.P, LDLSymbolic.pinv)
  end
  return LDLFactorization(SparseMatrixCSC{T,Ti}(LDLSymbolic.n, LDLSymbolic.n, LDLSymbolic.Lp, LDLSymbolic.Li, LDLSymbolic.Lx), LDLSymbolic.D, LDLSymbolic.P)
end
