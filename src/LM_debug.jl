using SparseArrays
using SolverTools
using NLPModels
using AMD
include("lma_aux.jl")
include("ldl_aux.jl")
include("qr_aux.jl")

"""
Implementation of Levenberg Marquardt algorithm for NLSModels
Solves min 1/2 ||r(x)||² where r is a vector of residuals
"""
function Levenberg_Marquardt(model :: AbstractNLSModel,
                             facto :: Symbol;
                             x :: AbstractVector=copy(model.meta.x0),
                             restol=100*sqrt(eps(eltype(x))),
                             satol=sqrt(eps(eltype(x))), srtol=sqrt(eps(eltype(x))),
                             oatol=0*sqrt(eps(eltype(x))), ortol=0*sqrt(eps(eltype(x))),
                             atol=100*sqrt(eps(eltype(x))), rtol=100*sqrt(eps(eltype(x))),
                             νd=eltype(x)(3), νm=eltype(x)(3), λ=eltype(x)(1.5), δd=eltype(x)(2),
                             ite_max :: Int=500)

  # @show model

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
  b = similar(r, (model.nls_meta.nequ + model.meta.nvar,))
  b[1 : model.nls_meta.nequ] .= -r
  xr = similar(b)

  # Initialize J in the format J[rows[k], cols[k]] = vals[k]
  rows = Vector{Int}(undef, model.nls_meta.nnzj)
  cols = Vector{Int}(undef, model.nls_meta.nnzj)
  jac_structure_residual!(model, rows, cols)
  vals = jac_coord_residual(model, x)

  if facto == :QR
    # Initialize A = [J; √λI] as a sparse matrix
    rows_A = vcat(rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    cols_A = vcat(cols, collect(1 : model.meta.nvar))
    vals_A = vcat(vals, fill(sqrt(λ), model.meta.nvar))
    A = sparse(rows_A, cols_A, vals_A)
    # Compute Jᵀr
    Jtr = transpose(A[1 : model.nls_meta.nequ, :]) * r

  elseif facto == :LDL
    # Initialize A = [[I J]; [Jᵀ - λI]] as sparse upper-triangular matrix
    cols_J = copy(cols)
    cols .+= model.nls_meta.nequ
    rows_A = vcat(collect(1 : model.nls_meta.nequ), rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    cols_A = vcat(collect(1 : model.nls_meta.nequ), cols, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar))
    vals_A = vcat(fill(1.0, model.nls_meta.nequ), vals, fill(-λ, model.meta.nvar))
    A = sparse(rows_A, cols_A, vals_A)
    # Compute permutation and perform symbolic analysis of A
    P = amd(A)
    ldl_symbolic = ldl_analyse(A, P, upper=true, n=model.meta.nvar + model.nls_meta.nequ)
    # Compute Jᵀr
    Jtr = transpose(A[1 : model.nls_meta.nequ, model.nls_meta.nequ + 1 : end]) * r
  end

  local norm_δ, δr2
  norm_Jtr = norm(Jtr)
  ϵ_first_order = atol + rtol * norm_Jtr
  old_obj = obj

  # Stopping criteria
  small_step = false
  first_order = norm_Jtr < ϵ_first_order
  small_residual = norm_r < restol
  small_obj_change =  false
  tired = iter > ite_max
  fail = false
  status = :unknown

  @info log_header([:iter, :f, :Δf, :dFeas, :λ, :δ, :ρ, :status], [Int, T, T, T, T, T, T, String],
                   hdr_override = Dict(:f => "f(x)", :dFeas => "‖Jᵀr‖", :δ => "‖δ‖"))

  while !(small_step || first_order || small_residual || small_obj_change || tired || fail)
    if facto == :QR
      # Solve min ‖[J √λI] δ + [r 0]‖² with QR factorization

      QR = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_AMD)
      δ, δr = solve_qr!(model.nls_meta.nequ + model.meta.nvar, model.meta.nvar, xr, b, QR.Q, QR.R, QR.prow, QR.pcol)
      # QR = qr(A)
      # δ = QR \ b

      # δ_correct = A \ b
      # δ .= δ_correct

      δr2 = norm(A[1 : model.nls_meta.nequ, :] * δ + r)^2 / 2

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
      # λ *= νm
      λ = max(λ, norm_δ) * νm

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

    iter += 1
    tired = iter > ite_max
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

  return GenericExecutionStats(status,
                               model,
                               solution = x,
                               objective = obj,
                               iter = iter,
                               elapsed_time = elapsed_time,
                               dual_feas = norm_Jtr)
end

# include("BALNLPModels.jl")
# BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :Metis, :None)
# print("\n ------------ \nStats : \n", stats)
