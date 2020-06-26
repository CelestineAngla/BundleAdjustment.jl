using JuMP
using Ipopt
using LinearAlgebra
using CodecBzip2


"""
Read the .txt.bzip2 file in Data/filename and extract the data,
returns the matrices observed 2D points, cameras and points
and the vectors of camera indices and points indices
"""
function readfile_JuMP(filename::String)
    filepath = joinpath(@__DIR__, "..", "Data", filename)
    f = Bzip2DecompressorStream(open(filepath))
    ncams, npnts, nobs = map(x -> parse(Int, x), split(readline(f)))

    @info "$filename: reading" ncams npnts nobs

    cam_indices = Vector{Int}(undef, nobs)
    pnt_indices = Vector{Int}(undef, nobs)
    pt2d = Matrix(undef, nobs, 2)

    # read nobs lines of the form
    # cam_index point_index xcoord ycoord
    for i = 1 : nobs
      cam, pnt, x, y = split(readline(f))
      cam_indices[i] = parse(Int, cam) + 1  # make indices start at 1
      pnt_indices[i] = parse(Int, pnt) + 1
      pt2d[i, 1] = parse(Float64, x)
      pt2d[i, 2] = parse(Float64, y)
    end
    # read 9 camera parameters, one per line, for each camera
    cam_params = Matrix(undef, ncams, 9)
    for i = 1 : ncams
      for j = 1 : 9
        cam_params[i, j] = parse(Float64, readline(f))
      end
    end
    # read npts 3d points, one coordinate per line
    pt3d = Matrix(undef, npnts, 3)
    for i = 1 : npnts
      for j = 1 : 3
        pt3d[i, j] = parse(Float64, readline(f))
      end
    end
    close(f)
    return cam_indices, pnt_indices, pt2d, cam_params, pt3d
end


"""
Computes the rotated vector from the original vector x and the 3D Rodrigues vector r
xᵣₒₜ = cos(θ) x+ sin(θ) k × x + (1 - cos(θ)) (k.x) k
with θ = ||r|| and k = r/θ and K the "cross product matrix" of k
"""
function Rodrigues_rotation(r, x)
    θ = sqrt(dot(r,r))
    k = r / θ
    return cos(θ) * x + sin(θ) * cross(k, x) + (1 - cos(θ)) * dot(k, x) * k
end


"""
Computes a scaling factor to undo the radial distortion
r(p) = 1.0 + k₁||p||² + k₂||p||⁴
"""
function scaling_factor(point, k1, k2)
    sq_norm_point = dot(point, point)
    return 1.0 + k1*sq_norm_point + k2*sq_norm_point^2
end


"""
Projection of point X on the given camera (r, t, f, k1, k2)
P₁ = RX + T
P₂ = -(P₁.x P₁.y)/P₁.z
P = f ⨯ r(P₂) ⨯ P₂
"""
function projection(x, y, z, rx, ry, rz, tx, ty, tz, f, k1, k2)
    P1 = Rodrigues_rotation([rx, ry, rz], [x; y; z]) + [tx, ty, tz]
    P2 = - [P1[1,1] P1[2,1]]/P1[3,1]
    return f*scaling_factor(P2, k1, k2)*P2
end


"""
Square residual function
||proj(X,C) - X_obs||²
"""
function sq_residuals(x, y, z, rx, ry, rz, tx, ty, tz, f, k1, k2, obs_x, obs_y)
    return sum((projection(x, y, z, rx, ry, rz, tx, ty, tz, f, k1, k2) - [obs_x obs_y])[k]^2 for k=1:2)
end


"""
Residual function
||proj(X,C) - X_obs||
"""
function residuals(x, y, z, rx, ry, rz, tx, ty, tz, f, k1, k2, obs_x, obs_y)
    return sqrt(sq_residuals(x, y, z, rx, ry, rz, tx, ty, tz, f, k1, k2, obs_x, obs_y))
end



"""
Builds the direct bundle adjustment optimization model
min 0.5 ∑ ||proj(X,C) - X_obs||²
"""
function direct_model(cam_indices::Vector{Int}, pnt_indices::Vector{Int}, pt2d::Matrix, cam_params_init::Matrix, pt3d_init::Matrix)
    nb_obs = size(pt2d, 1)
    nb_cameras = size(cam_params_init, 1)
    nb_points = size(pt3d_init, 1)

    model = Model(with_optimizer(Ipopt.Optimizer))
    @variable(model, cam_params[i=1:nb_cameras, j=1:9], start=cam_params_init[i,j])
    @variable(model, pt3d[i=1:nb_points, j=1:3], start=pt3d_init[i,j])
    register(model, :sqrt, 1, sqrt, autodiff=true)
    register(model, :sq_residuals, 14, sq_residuals, autodiff=true)

    @NLobjective(model, Min, 0.5*sum(sq_residuals(pt3d[pnt_indices[k],1], pt3d[pnt_indices[k],2], pt3d[pnt_indices[k],3], cam_params[cam_indices[k],1], cam_params[cam_indices[k],2], cam_params[cam_indices[k],3], cam_params[cam_indices[k],4], cam_params[cam_indices[k],5], cam_params[cam_indices[k],6], cam_params[cam_indices[k],7], cam_params[cam_indices[k],8], cam_params[cam_indices[k],9], pt2d[k,1], pt2d[k,2])[k] for k = 1:nb_obs))

    optimize!(model)
    # @show value.(cameras);
    # @show value.(points);
    @show objective_value.(model);
    return value.(cameras), value.(points)
end


"""
Builds the bundle adjustment optimization model with residuals variables to force Ipopt to consider it as a least square problem
min 0.5 ∑ r²
under ||proj(X,C) - X_obs|| + r = 0
"""
function residual_model(cam_indices::Vector{Int}, pnt_indices::Vector{Int}, pt2d::Matrix, cam_params::Matrix, pt3d::Matrix)
    nobs = size(pt2d)[1]
    ncam = size(cam_params)[1]
    npts = size(pt3d)[1]

    model = Model(with_optimizer(Ipopt.Optimizer))
    @variable(model, cameras[i=1:ncam, j=1:9], start=cam_params[i,j])
    @variable(model, points[i=1:npts, j=1:3], start=pt3d[i,j])
    register(model, :residuals, 14, residuals, autodiff=true)
    r_init = zeros(Float64, nobs)
    for k = 1:nobs
        r_init[k] = - residuals(pt3d[pnt_indices[k],1], pt3d[pnt_indices[k],2], pt3d[pnt_indices[k],3], cam_params[cam_indices[k],1], cam_params[cam_indices[k],2], cam_params[cam_indices[k],3], cam_params[cam_indices[k],4], cam_params[cam_indices[k],5], cam_params[cam_indices[k],6], cam_params[cam_indices[k],7], cam_params[cam_indices[k],8], cam_params[cam_indices[k],9], pt2d[k,1], pt2d[k,2])
    end
    @variable(model, r[k=1:nobs], start=r_init[k])

    for k = 1:nobs
        @NLconstraint(model, residuals(points[pnt_indices[k],1], points[pnt_indices[k],2], points[pnt_indices[k],3], cameras[cam_indices[k],1], cameras[cam_indices[k],2], cameras[cam_indices[k],3], cameras[cam_indices[k],4], cameras[cam_indices[k],5], cameras[cam_indices[k],6], cameras[cam_indices[k],7], cameras[cam_indices[k],8], cameras[cam_indices[k],9], pt2d[k,1], pt2d[k,2]) + r[k] == 0)
    end
    @NLobjective(model, Min, 0.5*sum(r[k]^2 for k = 1:nobs))
    optimize!(model)
    # @show value.(cameras);
    # @show value.(points);
    @show objective_value.(model);
    return value.(cameras), value.(points)
end


# # Get cameras indices, point indices, points 2D, cameras and points 3D matrices for datasets
# cam_indices, pnt_indices, pt2d, cam_params, pt3d = readfile_JuMP("LadyBug/problem-49-7776-pre.txt.bz2")
#
# # Find optimal camera features and points
# cameras_opt, points_opt = direct_model(cam_indices, pnt_indices, pt2d, cam_params, pt3d)
# cameras_opt, points_opt = residual_model(cam_indices, pnt_indices, pt2d, cam_params, pt3d)
