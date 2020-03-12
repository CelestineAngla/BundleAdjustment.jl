include("ReadFiles.jl")
include("Model.jl")


# Get obs, cameras and points matrices for datasets
obs, cameras, points = readfile("LadyBug/problem-49-7776-pre.txt")

# Find optimal camera features and points
# cameras_opt, points_opt = direct_model(obs, cameras, points)
cameras_opt, points_opt = residual_model(obs, cameras, points)

# # Save them into a  file
# writedlm("Cameras.txt", cameras_opt)
# writedlm("Points.txt", points_opt)
