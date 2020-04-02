using NLPModels
using CUTEst
using BenchmarkTools
include("BALNLPModels.jl")

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
