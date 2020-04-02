include("ReadFiles.jl")
include("ModelJuMP.jl")
include("BALNLPModels.jl")
include("LevenbergMarquardt.jl")

# # Get cameras indices, point indices, points 2D, cameras and points 3D matrices for datasets
# cam_indices, pnt_indices, pt2d, cam_params, pt3d = readfile("LadyBug/problem-49-7776-pre.txt.bz2")
#
# # Find optimal camera features and points
# cameras_opt, points_opt = direct_model(cam_indices, pnt_indices, pt2d, cam_params, pt3d)
# cameras_opt, points_opt = residual_model(cam_indices, pnt_indices, pt2d, cam_params, pt3d)


BA49 = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
fr_BA49 = FeasibilityResidual(BA49)
x_opt = @time Levenberg_Marquardt(fr_BA49, fr_BA49.meta.x0, 1e-5, 1e-2, 100000)


# J = jac(BA49, BA49.meta.x0)
#
# for j = 1 : 23769
#     if J[1,j] != 0
#         print("\n", j, " ", J[1,j])
#         print("\n", j, " ", J[2,j]) 3*npnts + 9*(idx_cam - 1) + 1 : 3*npnts + 9*(idx_cam - 1) + 9
#     end
# end


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

# Compare my resiudals with the one from python

# filepath = joinpath(@__DIR__, "..", "residuals_python_BA49.txt")
# f = open(filepath)
# python_res = Vector(undef, BA49.meta.ncon)
# for i = 1 : BA49.meta.ncon
#     python_res[i] = parse(Float64, readline(f))
# end

# Compare my vector of resiudals with and the one from CUTEst
# cx = Vector{Float64}(undef, BA49.meta.ncon)
# cx = cons!(BA49, BA49.meta.x0, cx)
# cx_sort = sort(cx)

# print(norm(cx-python_res))

# function increase_x(x, p)
#     return x + p*x
# end

# function mean_relative_error(x, y)
#     nobs = length(x)/2
#     error_x = 0.0
#     error_y = 0.0
#     for k = 1 : Integer(nobs)
#         if abs(x[2*k] - y[2*k])/abs(x[2*k]) > 10^(-3)
#             print(2*k)
#         end
#         error_x += abs(x[2*k - 1] - y[2*k - 1])/abs(x[2*k - 1])
#         error_y += abs(x[2*k] - y[2*k])/abs(x[2*k])
#     end
#     error_x /= nobs
#     error_y /= nobs
#     error = (error_x + error_y)/2
#     return error, error_x, error_y
# end

# using CUTEst
# BA49_cutest = CUTEstModel("BA-L49")
#
# err = Vector{Float64}(undef, 11)
# err_x = Vector{Float64}(undef, 11)
# err_y = Vector{Float64}(undef, 11)
# ps = [-2.5 -1.7 -1.2 -0.7 -0.3 0.0 0.4 0.7 1.2 1.7 2.5]
#
# for k = 1 : 11
#     x = increase_x(BA49.meta.x0, ps[k])
#     cx = Vector{Float64}(undef, BA49.meta.ncon)
#     cx = cons!(BA49, x, cx)
#     cx_cutest = Vector{Float64}(undef, BA49.meta.ncon)
#     cx_cutest = cons(BA49_cutest, x)
#     err[k], err_x[k], err_y[k] = mean_relative_error(cx, cx_cutest)
# end



# using CUTEst
# BA49_cutest = CUTEstModel("BA-L49")
#
# x = BA49.meta.x0
#
# cx = Vector{Float64}(undef, BA49.meta.ncon)
# cx = cons!(BA49, x, cx)
# cx_cutest = Vector{Float64}(undef, BA49.meta.ncon)
# cx_cutest = cons(BA49_cutest, x)
#
# for k = 1 : length(cx)
#     if abs(cx[k] - cx_cutest[k])/abs(cx[k]) > 10^(1)
#         print("\n", k, " ", cx[k], " ", cx_cutest[k])
#     end
# end

# print(mean_relative_error(cx, cx_cutest))
#
# finalize(BA49_cutest)
