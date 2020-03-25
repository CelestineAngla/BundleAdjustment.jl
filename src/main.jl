include("ReadFiles.jl")
include("ModelJuMP.jl")
include("BALNLPModels.jl")
using SparseArrays


# # Get cameras indices, point indices, points 2D, cameras and points 3D matrices for datasets
# cam_indices, pnt_indices, pt2d, cam_params, pt3d = readfile("LadyBug/problem-49-7776-pre.txt.bz2")
#
# # Find optimal camera features and points
# cameras_opt, points_opt = direct_model(cam_indices, pnt_indices, pt2d, cam_params, pt3d)
# cameras_opt, points_opt = residual_model(cam_indices, pnt_indices, pt2d, cam_params, pt3d)


BA49 = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")

J = jac(BA49, BA49.meta.x0)

for j = 1 : 23769
    if J[1,j] != 0
        print("\n", j, " ", J[1,j])
        print("\n", j, " ", J[2,j])
    end
end


# err = jacobian_check(BA49, x=BA49.meta.x0)
# print(err)


# # Compare the structure my jacobian and of the one from CUTEst
#
# J = jac(BA49, BA49.meta.x0)
# nz = findnz(J)
#
# using CUTEst
# BA49_cutest = CUTEstModel("BA-L49")
# rows, cols = jac_structure(BA49_cutest)
# vals = jac_coord(BA49_cutest,BA49_cutest.meta.x0)
# nzc = findnz(sparse(rows, cols,vals))
# finalize(BA49_cutest)
#
# print("\n", norm(nz[1]-nzc[1]))
# print("\n", norm(nz[2]-nzc[2]))
