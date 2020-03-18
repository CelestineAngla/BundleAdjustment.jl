using JuMP
using Ipopt
using LinearAlgebra


"""
Computes the rotated vector from the original vector x and the 3D Rodrigues vector r
xᵣₒₜ = cos(θ) x+ sin(θ) k × x + (1 - cos(θ)) (k.x) k
with θ = ||r|| and k = r/θ and K the "cross product matrix" of k
"""
function Rodrigues_rotation(r, x)
    θ = norm(r)
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
function direct_model(obs::Array{Float64,2}, cameras_init::Array{Float64,2}, points_init::Array{Float64,2})
    nb_obs = size(obs)[1]
    nb_cameras = size(cameras_init)[1]
    nb_points = size(points_init)[1]

    model = Model(with_optimizer(Ipopt.Optimizer))
    @variable(model, cameras[i=1:nb_cameras, j=1:9], start=cameras_init[i,j])
    @variable(model, points[i=1:nb_points, j=1:3], start=points_init[i,j])
    register(model, :sqrt, 1, sqrt, autodiff=true)
    register(model, :sq_residuals, 14, sq_residuals, autodiff=true)

    @NLobjective(model, Min, 0.5*sum(sq_residuals(points[Integer(obs[k,2]+1),1], points[Integer(obs[k,2]+1),2], points[Integer(obs[k,2]+1),3], cameras[Integer(obs[k,1]+1),1], cameras[Integer(obs[k,1]+1),2], cameras[Integer(obs[k,1]+1),3], cameras[Integer(obs[k,1]+1),4], cameras[Integer(obs[k,1]+1),5], cameras[Integer(obs[k,1]+1),6], cameras[Integer(obs[k,1]+1),7], cameras[Integer(obs[k,1]+1),8], cameras[Integer(obs[k,1]+1),9], obs[k,3], obs[k,4])[k] for k = 1:nb_obs))
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
