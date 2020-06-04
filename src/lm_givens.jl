using SparseArrays
using SolverTools
using NLPModels
using AMD
using Metis
include("lma_aux.jl")
include("ldl_aux.jl")
include("qr_aux.jl")

"""
Implementation of Levenberg Marquardt algorithm for NLSModels
Solves min 1/2 ||r(x)||² where r is a vector of residuals
"""
function Levenberg_Marquardt(model :: AbstractNLSModel,
                             facto :: Symbol,
                             perm :: Symbol;
                             x :: AbstractVector=copy(model.meta.x0),
                             restol=eltype(x)(eps(eltype(x))^(1/3)),
                             satol=sqrt(eps(eltype(x))), srtol=sqrt(eps(eltype(x))),
                             oatol=sqrt(eps(eltype(x))), ortol=eltype(x)(eps(eltype(x))^(1/3)),
                             atol=sqrt(eps(eltype(x))), rtol=eltype(x)(eps(eltype(x))^(1/3)),
                             νd=eltype(x)(3), νm=eltype(x)(3), λ=eltype(x)(30), δd=eltype(x)(2),
                             ite_max :: Int=200, max_time :: Int=3600)

  @info model
  @info "Parameters of the solver:\n" facto perm νd νm λ ite_max
  @info "Tolerances:\n" restol satol srtol oatol ortol atol rtol

  start_time = time()
  elapsed_time = 0.0
  iter = 0
  x_suiv = similar(x)
  T = eltype(x)

  # Initialize residuals
  r = residual(model, x)
  norm_r = norm(r)
  obj = norm_r^2 / 2
  r_suiv = copy(r)

  # Initialize b = [r; 0]
  b = Vector{T}(undef, model.nls_meta.nequ + model.meta.nvar)
  b[1 : model.nls_meta.nequ] .= -r
  b[model.nls_meta.nequ + 1 : end] .= 0
  xr = similar(b)

  # Initialize J in the format J[rows[k], cols[k]] = vals[k]
  rows = Vector{Int}(undef, model.nls_meta.nnzj)
  cols = Vector{Int}(undef, model.nls_meta.nnzj)
  jac_structure_residual!(model, rows, cols)
  vals = jac_coord_residual(model, x)

  # Compute Jᵀr and λ
  Jtr = mul_sparse(cols, rows, vals, r, model.nls_meta.nnzj, model.meta.nvar)
  norm_Jtr = norm(Jtr)
  λ = T(max(λ, 1e10 / norm_Jtr))

  if facto == :QR
    # Initialize A = [J; √λI] as a sparse matrix
    rows_A = vcat(rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    cols_A = vcat(cols, collect(1 : model.meta.nvar))
    vals_A = vcat(vals, fill(sqrt(λ), model.meta.nvar))
    A = sparse(rows_A, cols_A, vals_A)

    # Givens version
    col_norms = ones(model.meta.nvar)
    QR_J = qr(A[1 : model.nls_meta.nequ, :])
    R = similar(QR_J.R)
    nnz_R = nnz(QR_J.R)
    # Rt = sparse(QR_J.R')
    G_list = Vector{LinearAlgebra.Givens{Float64}}(undef, Int(model.meta.nvar*(model.meta.nvar + 1)/2))
    news = zeros(T, model.meta.nvar)
    Prow = vcat(QR_J.prow, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))

  elseif facto == :LDL
    # Initialize A = [[I J]; [Jᵀ - λI]] as sparse upper-triangular matrix
    cols_J = copy(cols)
    cols .+= model.nls_meta.nequ
    rows_A = vcat(collect(1 : model.nls_meta.nequ), rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    cols_A = vcat(collect(1 : model.nls_meta.nequ), cols, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    vals_A = vcat(fill(1.0, model.nls_meta.nequ), vals, fill(-λ, model.meta.nvar))
    A = sparse(rows_A, cols_A, vals_A)
    # Compute permutation and perform symbolic analysis of A
    if perm == :AMD
      P = amd(A)
    elseif perm == :Metis
      P , _ = Metis.permutation(A' + A)
    end
    ldl_symbolic = ldl_analyse(A, P, upper=true, n=model.meta.nvar + model.nls_meta.nequ)
  end

  local norm_δ, δr2
  ϵ_first_order = atol + rtol * norm_Jtr
  old_obj = obj

  # Stopping criteria
  small_step = false
  first_order = norm_Jtr < ϵ_first_order
  small_residual = norm_r < restol
  small_obj_change =  false
  tired = iter > ite_max || elapsed_time > max_time
  fail = false
  status = :unknown

  @info log_header([:iter, :f, :Δf, :dFeas, :λ, :δ, :ρ, :status], [Int, T, T, T, T, T, T, String],
                   hdr_override = Dict(:f => "f(x)", :dFeas => "‖Jᵀr‖", :δ => "‖δ‖"))

  while !(small_step || first_order || small_residual || small_obj_change || tired || fail)
    iter += 1

    if facto == :QR
      # Givens version
      @time begin
        R .= QR_J.R
      end
      @time begin
        nnz_R = nnz(R)
      end
      print("\n nnz_R : ", nnz_R)
      @time begin
        Rt = sparse(R')
      end
      @time begin
        counter = fullQR_givens!(R, Rt, G_list, news, sqrt(λ), model.meta.nvar, model.nls_meta.nequ, nnz_R)
      end
      @time begin
        δ, δr = solve_qr2!(model.nls_meta.nequ + model.meta.nvar, model.meta.nvar, xr, b, QR_J.Q, R, Prow, QR_J.pcol, counter, G_list)
      end
      print("\n", norm(A * δ - b), " ", norm(δr))
      @time begin
        news .= 0
      end

      # # Solve min ‖[J √λI] δ + [r 0]‖² with QR factorization
      # if perm == :AMD
      #   QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_AMD)
      # elseif perm == :Metis
      #   QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_METIS)
      # end
      # δ, δr = solve_qr!(model.nls_meta.nequ + model.meta.nvar, model.meta.nvar, xr, b, QR.Q, QR.R, QR.prow, QR.pcol)
      δr2 = norm(A[1 : model.nls_meta.nequ, :] * δ + r)^2 / 2
      δc = A \ b
      print("\n", norm(δ - δc))

    elseif facto == :LDL
      # Solve [[I J]; [Jᵀ - λI]] X = [-r; 0] with LDL factorization
      LDLT = ldl_factorize(A, ldl_symbolic, true)
      xr .= b
      ldl_solve!(model.nls_meta.nequ + model.meta.nvar, xr, LDLT.L.colptr, LDLT.L.rowval, LDLT.L.nzval, LDLT.D, P)
      δr = xr[1 : model.nls_meta.nequ]
      δ = xr[model.nls_meta.nequ + 1 : end]
      δr2 = norm(δr)^2 / 2
    end

    # Check model decrease
    if δr2 > obj
      @error "‖δr‖² > ‖r‖²" δr2 obj
      fail = true
      continue
    end

    norm_δ = norm(δ)
    x_suiv .=  x + δ
    residual!(model, x_suiv, r_suiv)
    norm_rsuiv = norm(r_suiv)
    obj_suiv = norm_rsuiv^2 / 2

    # Step not accepted : d(||r||²) < 1e-4 (||Jδ + r||² - ||r||²)
    pred = obj - δr2       # predicted reduction
    ared = obj - obj_suiv  # actual reduction
    step_accepted = ared ≥ 1e-4 * pred
    step_accepted_str = step_accepted ? "acc" : "rej"
    @info log_row((iter, obj, old_obj - obj, norm_Jtr, λ, norm_δ, ared / pred, step_accepted_str))

    if !step_accepted
      # Update λ
      λ = max(λ, 1 / norm_δ) * νm

      # Update A
      if facto == :QR
        vals_A[model.nls_meta.nnzj + 1 : end] .= sqrt(λ)
        A = sparse(rows_A, cols_A, vals_A)

      # Update A
      elseif facto == :LDL
        vals_A[model.nls_meta.nequ + model.nls_meta.nnzj + 1 : end] .= -λ
        A = sparse(rows_A, cols_A, vals_A)
      end

    else
      # Update λ and x
      λ /= νd
      if ared ≥ 0.9 * pred  # very successful step
        λ /= νd
      end
      λ = max(1.0e-8, λ)
      x .= x_suiv

      # Update J
      jac_coord_residual!(model, x, vals)

      # Update r and b
      old_obj = obj
      r .= r_suiv
      b[1 : model.nls_meta.nequ] .= -r
      norm_r = norm_rsuiv
      obj = obj_suiv

      # Update A and Jtr
      if facto == :QR
        @views vals_A[1 : model.nls_meta.nnzj] = vals
        vals_A[model.nls_meta.nnzj + 1 : end] .= sqrt(λ)
        A = sparse(rows_A, cols_A, vals_A)

        # Givens version
        QR_J = qr(A[1 : model.nls_meta.nequ, :])

        mul_sparse!(Jtr, cols, rows, vals, r, model.nls_meta.nnzj)

      elseif facto == :LDL
        @views vals_A[model.nls_meta.nequ + 1 :  model.nls_meta.nequ + model.nls_meta.nnzj] = vals
        vals_A[model.nls_meta.nequ + model.nls_meta.nnzj + 1 : end] .= -λ
        A = sparse(rows_A, cols_A, vals_A)

        mul_sparse!(Jtr, cols_J, rows, vals, r, model.nls_meta.nnzj)
      end

      # Update the stopping criteria
      norm_Jtr = norm(Jtr)
      small_step = norm_δ < satol + srtol * norm(x)
      first_order = norm_Jtr < ϵ_first_order
      small_residual = norm_r < restol
      # ‖rₖ₋₁‖²/2 - ‖rₖ‖²/2 < oatol + ortol * ‖rₖ₋₁‖²/2
      small_obj_change =  old_obj - obj < oatol + ortol * old_obj
    end

    tired = iter > ite_max || elapsed_time > max_time
  end

  # fail || (@info log_row((iter, obj, old_obj - obj, norm_Jtr, λ, norm_δ, δr2, δr2 - obj, "")))

  if small_step
    status = :small_step
  elseif first_order
    status = :first_order
  elseif small_residual
    status = :small_residual
  elseif small_obj_change
    status = :acceptable
  elseif fail
    status = :neg_pred
  elseif tired
    status = :max_iter
  end

  elapsed_time = time() - start_time

  stats = GenericExecutionStats(status,
                               model,
                               solution = x,
                               objective = obj,
                               iter = iter,
                               elapsed_time = elapsed_time,
                               dual_feas = norm_Jtr)
  @info stats
  return stats
end

include("BALNLPModels.jl")
BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
fr_BA = FeasibilityResidual(BA)
stats = Levenberg_Marquardt(fr_BA, :QR, :AMD)
