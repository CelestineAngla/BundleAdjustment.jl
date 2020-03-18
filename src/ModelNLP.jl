using NLPModels

function projection!(p3, r, t, f, k, r2)
  θ = norm(r)
  kx, ky, kz = r / θ
  K = [0.0 -kz ky; kz 0.0 -kx; -ky kx 0.0]
  Kp3 = K * p3
  P1 = cos(θ) * p3 + sin(θ) * Kp3 + (1 - cos(θ)) * K * Kp3 + t
  P2 = -P1[1:2] / P1[3]
  r2[:] = f * scaling_factor(P2, k) * P2
  return r2
end

projection!(x, r2) = projection!(x[1:3], x[4:6], x[7:9], x[10], x[11:12], r2)

function residuals!(cam_indices, pnt_indices, xs, r)
  nobs = length(cam_indices)
  for k = 1 : nobs
    cam_index = cam_indices[k]
    pnt_index = pnt_indices[k]
    x = xs[(pnt_index - 1) * 12 + 1 : pnt_index * 12]
    projection!(x, r[2 * k - 1 : 2 * k])
  end
  return r
end


"""
Represent a bundle adjustement problem in the form

    minimize    0
    subject to  F(x) = 0,

where `F(x)` is the vector of residuals.
"""
mutable struct BALNLPModel <: AbstractNLPModel
  cams
  pnts
  pt2d
  cam_params
  pt3d
  meta :: NLPModelMeta
  counters :: Counters
end


function BALNLPModel(filename::AbstractString)
  cams, pnts, pt2d, cam_params, pt3d = read_bal(filename)

  # variables: 9 parameters per camera + 3 coords per 3d point
  ncams = size(cam_params, 1)
  npnts = size(pt3d, 1)
  nvar = 9 * ncams + 3 * npnts

  # number of residuals: one residual per 2d point
  nobs = size(pt2d, 1)
  ncon = 2 * nobs

  # # determine sparsity pattern and number of nonzeros in Jacobian
  # input = rand(nvar)
  # output = rand(ncon)
  # resid!(x, r) = residuals!(cams, pnts, x, r)
  # pattern = sparsity!(resid!, input, output)
  # nnzj = nnz(pattern)

  @info "BALNLPModel $filename" nvar ncon nnzj
end

function NLPModels.jac_coord(nlp :: BALNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return ([1, 1], [1, 2], [-20 * x[1], 10.0])
end
