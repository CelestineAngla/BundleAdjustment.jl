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

    # x0 = [X₁, X₂, ..., Xnpnts, C₁, C₂, ..., Cncam]
    x0 = Vector{Float64}(undef, 3 * npnts + 9 * ncams)
    # read 9 camera parameters, one per line, for each camera
    for i = 1 : ncams
      for j = 1 : 7
        if j == 7
          # Cam_param = (rx, ry, rz, tx, ty, tz, k1, k2, f)
          x0[3 * npnts + 9 * (i - 1) + 9] = parse(Float64, readline(f))
          x0[3 * npnts + 9 * (i - 1) + 7] = parse(Float64, readline(f))
          x0[3 * npnts + 9 * (i - 1) + 8] = parse(Float64, readline(f))
        else
          x0[3 * npnts + 9 * (i - 1) + j] = parse(Float64, readline(f))
        end
      end
    end
    # read npts 3d points, one coordinate per line
    for k = 1 : 3 * npnts
      x0[k] = parse(Float64, readline(f))
    end


    close(f)

    return cam_indices, pnt_indices, pt2d, x0, ncams, npnts, nobs
end
