using NLPModels
using CUTEst
using BenchmarkTools

"""
Compare execution time between CUTEst residuals and jacobian and mine
"""


BA49 = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
finalize(BA49_cutest)
BA49_cutest = CUTEstModel("BA-L49")

@btime cons(BA49, BA49.meta.x0)
@btime jac(BA49, BA49.meta.x0)
rows, cols = @btime jac_structure(BA49)
vals = zeros(length(rows))
@btime jac_coord!(BA49, BA49.meta.x0, vals)

@btime cons(BA49_cutest, BA49_cutest.meta.x0)
@btime jac(BA49_cutest, BA49_cutest.meta.x0)
rows, cols = @btime jac_structure(BA49_cutest)
vals = zeros(length(rows))
@btime jac_coord!(BA49_cutest, BA49_cutest.meta.x0, vals)
