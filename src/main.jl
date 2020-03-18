include("ReadFiles.jl")
include("ModelJuMP.jl")
include("JacobianByHand.jl")


# Get cameras indices, point indices, points 2D, cameras and points 3D matrices for datasets
cam_indices, pnt_indices, pt2d, cam_params, pt3d = readfile("LadyBug/problem-49-7776-pre.txt.bz2")

J = BAJacobian(cam_indices, pnt_indices, pt2d, cam_params, pt3d)


# Find optimal camera features and points
# cameras_opt, points_opt = direct_model(cam_indices, pnt_indices, pt2d, cam_params, pt3d)
# cameras_opt, points_opt = residual_model(cam_indices, pnt_indices, pt2d, cam_params, pt3d)

# # Save them into a  file
# writedlm("Cameras.txt", cameras_opt)
# writedlm("Points.txt", points_opt)
