using JuMP
using Ipopt
using LinearAlgebra


"""
Creates the 3D rotation matrix associated to the 3D Rodrigues vector r
R = cos(θ)I₃ + sin(θ)K + (1 - cos(θ))kkᵀ
with θ = ||r|| and k = r/θ and K the "cross product matrix" of k
"""
function get_rotation_matrix(rodrigues_vect)
    angle = sqrt(sum(rodrigues_vect[k]^2 for k = 1:3))
    # angle = LinearAlgebra.norm(rodrigues_vect)
    kx, ky, kz = rodrigues_vect/angle
    Id = [[1 0 0]; [0 1 0]; [0 0 1]]
    K = [[0 -kz ky]; [kz 0 -kx]; [-ky kx 0]]
    kkT = [[kx*kx kx*ky kx*kz]; [ky*kx ky*ky ky*kz]; [kz*kx kz*ky kz*kz]]
    return Id*cos(angle) + sin(angle)*K + (1 - cos(angle))*kkT
end


"""
Computes a scaling factor to undo the radial distortion
r(p) = 1.0 + k₁||p||² + k₂||p||⁴
"""
function scaling_factor(point, k1, k2)
    sq_norm_point = point[1]^2 + point[2]^2
    return 1.0 + k1*sq_norm_point + k2*sq_norm_point^2
end


"""
Projection of point X on the given camera (r, t, f, k1, k2)
P₁ = RX + T
P₂ = -(P₁.x P₁.y)/P₁.z
P = f ⨯ r(P₂) ⨯ P₂
"""
function projection(x, y, z, rx, ry, rz, tx, ty, tz, f, k1, k2)
    X = [x; y; z]
    R = get_rotation_matrix([rx, ry, rz])
    P1 = R*X + [tx, ty, tz]
    P2 = -[P1[1,1] P1[2,1]]/P1[3,1]
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
function residual_model(obs::Array{Float64,2}, cameras_init::Array{Float64,2}, points_init::Array{Float64,2})
    nb_obs = size(obs)[1]
    nb_cameras = size(cameras_init)[1]
    nb_points = size(points_init)[1]

    model = Model(with_optimizer(Ipopt.Optimizer))
    @variable(model, cameras[i=1:nb_cameras, j=1:9], start=cameras_init[i,j])
    @variable(model, points[i=1:nb_points, j=1:3], start=points_init[i,j])
    register(model, :residuals, 14, residuals, autodiff=true)
    r_init = zeros(Float64, nb_obs)
    for k = 1:nb_obs
        r_init[k] = - residuals(points_init[Integer(obs[k,2]+1),1], points_init[Integer(obs[k,2]+1),2], points_init[Integer(obs[k,2]+1),3], cameras_init[Integer(obs[k,1]+1),1], cameras_init[Integer(obs[k,1]+1),2], cameras_init[Integer(obs[k,1]+1),3], cameras_init[Integer(obs[k,1]+1),4], cameras_init[Integer(obs[k,1]+1),5], cameras_init[Integer(obs[k,1]+1),6], cameras_init[Integer(obs[k,1]+1),7], cameras_init[Integer(obs[k,1]+1),8], cameras_init[Integer(obs[k,1]+1),9], obs[k,3], obs[k,4])
    end
    @variable(model, r[k=1:nb_obs], start=r_init[k])

    for k = 1:nb_obs
        @NLconstraint(model, residuals(points[Integer(obs[k,2]+1),1], points[Integer(obs[k,2]+1),2], points[Integer(obs[k,2]+1),3], cameras[Integer(obs[k,1]+1),1], cameras[Integer(obs[k,1]+1),2], cameras[Integer(obs[k,1]+1),3], cameras[Integer(obs[k,1]+1),4], cameras[Integer(obs[k,1]+1),5], cameras[Integer(obs[k,1]+1),6], cameras[Integer(obs[k,1]+1),7], cameras[Integer(obs[k,1]+1),8], cameras[Integer(obs[k,1]+1),9], obs[k,3], obs[k,4]) + r[k] == 0)
    end
    @NLobjective(model, Min, 0.5*sum(r[k]^2 for k = 1:nb_obs))
    optimize!(model)
    # @show value.(cameras);
    # @show value.(points);
    @show objective_value.(model);
    return value.(cameras), value.(points)
end

# cos(angle)*x + sin(angle)*cross(k,x) + (1-cos(angle))*dot(k,x)*k

# k = [1, 1, 1]
# x = [2.5, -6.1, 0.0]
# print(dot(k,x))
# print(cross(k,x))
