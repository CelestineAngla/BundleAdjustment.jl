# Extracting the datasets given at https://grail.cs.washington.edu/projects/bal/


"""
Read the .txt file in Data/filename and extract the data,
returns the matrices observed points, cameras and points
"""
function readfile(filename::String)
    filepath = joinpath(@__DIR__, "..", "Data", filename)
    f = open(filepath)
    lines = readlines(f)
    nb_cameras, nb_points, nb_obs = [parse(Int64, x) for x in split(lines[1])]

    # The obs matrix contains, for each observation,
    # the camera index, the point index, the coordinates (x,y) of the observed point
    obs = zeros(Float64, (nb_obs, 4))
    for i = 1:nb_obs
        obs[i,:] = [parse(Float64, x) for x in split(lines[i + 1])]
    end

    # The cameras matrix containes (r,t,f,k1,k2) for each camera
    # Where r = (rx,ry,rz) Rodrigues vector of the camera rotation
    # t = (tx,ty,tz) the coordinates of the translation vector of the camera
    # f the focal distance and (k1,k2) radial distortion parameters
    cameras = zeros(Float64, (nb_cameras, 9))
    for i = 1:nb_cameras
        for j = 1:9
            cameras[i,j] = parse(Float64, lines[(i-1)*9 + j + nb_obs+1])
        end
    end

    # The points matrix contains the actual 3D coordinates of all the points
    points = zeros(Float64, (nb_points, 3))
    for i = 1:nb_points
        for j = 1:3
            points[i,j] = parse(Float64, lines[(i-1)*3 + j + nb_obs+nb_cameras*9+1])
        end
    end

    close(f)
    return obs, cameras, points
end



# dir_path = joinpath(@__DIR__, "..", "Data")
#
# foreach(readdir(dir_path)) do d
#     println("----------  \n")
#     println(d)
#     foreach(readdir(joinpath("Data", d))) do f
#         println(f)
#     end
# end
