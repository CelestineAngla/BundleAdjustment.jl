using CUTEst
using BenchmarkTools
using ProfileView
include("BALNLPModels.jl")

# fetch_sif_problems()

"""
Compare execution time between CUTEst residuals and jacobian and mine
"""


BA49 = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")

BA49_cutest = CUTEstModel("BA-L49")

@btime cons(BA49, BA49.meta.x0)
@btime jac(BA49, BA49.meta.x0)
rows = Vector{Int}(undef, BA49.meta.nnzj)
cols = Vector{Int}(undef, BA49.meta.nnzj)
@btime jac_structure!(BA49, rows, cols)
vals = zeros(length(rows))
@btime jac_coord!(BA49, BA49.meta.x0, vals)

@btime cons(BA49_cutest, BA49_cutest.meta.x0)
@btime jac(BA49_cutest, BA49_cutest.meta.x0)
rows = Vector{Int}(undef, BA49_cutest.meta.nnzj)
cols = Vector{Int}(undef, BA49_cutest.meta.nnzj)
@btime jac_structure!(BA49_cutest, rows, cols)
vals = zeros(length(rows))
@btime jac_coord!(BA49_cutest, BA49_cutest.meta.x0, vals)

finalize(BA49_cutest)



# rows = Vector{Int}(undef, BA49.meta.nnzj)
# cols = Vector{Int}(undef, BA49.meta.nnzj)
# @profview jac_structure!(BA49, rows, cols)




# # Compare my resiudals with the one from python
#
# filepath = joinpath(@__DIR__, "..", "residuals_python_BA49.txt")
# f = open(filepath)
# python_res = Vector(undef, BA49.meta.ncon)
# for i = 1 : BA49.meta.ncon
#     python_res[i] = parse(Float64, readline(f))
# end
#
# cx = Vector{Float64}(undef, BA49.meta.ncon)
# cx = cons!(BA49, BA49.meta.x0, cx)
#
# print(norm(cx-python_res))





# # Compare the structure my jacobian and of the one from CUTEst
#
# BA49 = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
# J = jac(BA49, BA49.meta.x0)
# nz = findnz(J)
#
# BA49_cutest = CUTEstModel("BA-L49")
# rows, cols = jac_structure(BA49_cutest)
# vals = jac_coord(BA49_cutest,BA49_cutest.meta.x0)
# nzc = findnz(sparse(rows, cols,vals))
# finalize(BA49_cutest)
#
# print("\n", norm(nz[1]-nzc[1]))
# print("\n", norm(nz[2]-nzc[2]))




# err = jacobian_check(BA49, x=BA49.meta.x0)
# print(err)




# # Compare my residuals with the ones from CUTEst
#
# function increase_x(x, p)
#     return x + p*x
# end
#
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
#
# BA49 = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
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
