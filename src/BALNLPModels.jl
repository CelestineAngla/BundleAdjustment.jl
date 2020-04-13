using NLPModels
using LinearAlgebra
using .Threads
import NLPModels: increment!
include("ReadFiles.jl")
include("JacobianByHand.jl")


function scaling_factor(point, k1, k2)
    sq_norm_point = dot(point, point)
    return 1.0 + k1*sq_norm_point + k2*sq_norm_point^2
end


function projection!(p3, r, t, k1, k2, f, r2)
  θ = norm(r)
  k = r / θ
  P1 = cos(θ) * p3 + sin(θ) * cross(k, p3) + (1 - cos(θ)) * dot(k, p3) * k + t
  P2 = -P1[1:2] / P1[3]
  r2 .= f * scaling_factor(P2, k1, k2) * P2
  return r2
end


projection!(x, c, r2) = projection!(x, c[1:3], c[4:6], c[7], c[8], c[9], r2)


function residuals!(cam_indices, pnt_indices, xs, r, nobs, npts)
  for k = 1 : nobs
    cam_index = cam_indices[k]
    pnt_index = pnt_indices[k]
    @views x = xs[(pnt_index - 1) * 3 + 1 : (pnt_index - 1) * 3 + 3]
    @views c = xs[3*npts + (cam_index - 1) * 9 + 1 : 3*npts + (cam_index - 1) * 9 + 9]
    @views projection!(x, c, r[2 * k - 1 : 2 * k])
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
  meta :: NLPModelMeta
  counters :: Counters
  cams_indices
  pnts_indices
  pt2d
  nobs
  npnts
  ncams
end


function BALNLPModel(filename::AbstractString)
  cams_indices, pnts_indices, pt2d, x0, ncams, npnts, nobs = readfile(filename)

  # variables: 9 parameters per camera + 3 coords per 3d point
  nvar = 9 * ncams + 3 * npnts
  # number of residuals: two residuals per 2d point
  ncon = 2 * nobs

  meta = NLPModelMeta(nvar, ncon=ncon, x0=x0, lcon=fill(0.0,ncon), ucon=fill(0.0,ncon), nnzj=2*nobs*12, name=filename)

  @info "BALNLPModel $filename" nvar ncon
  return BALNLPModel(meta, Counters(), cams_indices, pnts_indices, pt2d, nobs, npnts, ncams)
end


NLPModels.obj(model::BALNLPModel, x::AbstractVector) = 0.0


NLPModels.grad!(model::BALNLPModel, x::AbstractVector, g::AbstractVector) = fill!(g, 0)


function NLPModels.cons!(nlp :: BALNLPModel, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  residuals!(nlp.cams_indices, nlp.pnts_indices, x, cx, nlp.nobs, nlp.npnts)
  cx .-= nlp.pt2d
  return cx
end


function NLPModels.jac_structure!(nlp :: BALNLPModel, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
  increment!(nlp, :neval_jac)
  nobs = nlp.nobs
  npnts_3 = 3 * nlp.npnts

  for k = 1 : nobs
    idx_obs = (k - 1) * 24
    idx_cam = npnts_3 + 9* (nlp.cams_indices[k] - 1)
    idx_pnt = 3 * (nlp.pnts_indices[k] - 1)

    # Only the two rows corresponding to the observation k are not empty
    p = 2 * k
    @views fill!(rows[idx_obs + 1 : idx_obs + 12], p - 1)
    @views fill!(rows[idx_obs + 13 : idx_obs + 24], p)

    # 3 columns for the 3D point observed
    @inbounds cols[idx_obs + 1 : idx_obs + 3] = idx_pnt + 1 : idx_pnt + 3
    # 9 columns for the camera
    @inbounds cols[idx_obs + 4 : idx_obs + 12] = idx_cam + 1 : idx_cam + 9
    # 3 columns for the 3D point observed
    @inbounds cols[idx_obs + 13 : idx_obs + 15] = idx_pnt + 1 : idx_pnt + 3
    # 9 columns for the camera
    @inbounds cols[idx_obs + 16 : idx_obs + 24] = idx_cam + 1 : idx_cam + 9
  end

  return rows, cols
end


function NLPModels.jac_coord!(nlp :: BALNLPModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  nobs = nlp.nobs
  npnts = nlp.npnts
  denseJ = Matrix{Float64}(undef, 2, 12)
  JP1_mat = zeros(6, 12)
  JP1_mat[1, 7], JP1_mat[2, 8], JP1_mat[3, 9], JP1_mat[4, 10], JP1_mat[5, 11], JP1_mat[6, 12] = 1, 1, 1, 1, 1, 1
  JP2_mat = zeros(5, 6)
  JP2_mat[3, 4], JP2_mat[4, 5], JP2_mat[5, 6] = 1, 1, 1
  JP3_mat = Matrix{Float64}(undef, 2, 5)
  for k = 1 : nobs
    idx_cam = nlp.cams_indices[k]
    idx_pnt = nlp.pnts_indices[k]
    @views X = x[(idx_pnt - 1) * 3 + 1 : (idx_pnt - 1) * 3 + 3] # 3D point coordinates
    @views C = x[3*npnts + (idx_cam - 1) * 9 + 1 : 3*npnts + (idx_cam - 1) * 9 + 9] # camera parameters
    r = C[1:3]  # Rodrigues vector for the rotation
    t = C[4:6]  # translation vector
    k1, k2, f = C[7:9]  # focal length and radial distortion factors

    # denseJ = JP3∘P2∘P1 x JP2∘P1 x JP1
    p1 = P1(r, t, X)
    JP1!(JP1_mat, r, X)
    JP2!(JP2_mat, p1)
    JP3!(JP3_mat, P2(p1), f, k1, k2)
    mul!(denseJ, JP3_mat*JP2_mat, JP1_mat)

    # Feel vals with the values of denseJ = [[∂P.x/∂X ∂P.x/∂C], [∂P.y/∂X ∂P.y/∂C]]
    vals[(k-1)*24 + 1 : (k-1)*24 + 24] .= denseJ'[:]

  end
  return vals
end
