# Extracting the datasets given at https://grail.cs.washington.edu/projects/bal/
using CodecBzip2

"""
Read the .txt.bzip2 file in Data/filename and extract the data,
returns the matrices observed 2D points, cameras and points
and the vectors of camera indices and points indices
"""
function readfile(filename::String)
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
