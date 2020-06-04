using SolverTools
using DataFrames
include("BALNLPModels.jl")
# include("LevenbergMarquardt.jl")
include("lm.jl")

# Create a BALNLPModel from a dataset
BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-73-11032-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-138-19878-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-318-41628-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-460-56811-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-646-73584-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-810-88814-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-1031-110968-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-1235-129634-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-1514-147317-pre.txt.bz2")
# BA = BALNLPModel("LadyBug/problem-1723-156502-pre.txt.bz2")
# BA = BALNLPModel("Dubrovnik/problem-202-132796-pre.txt.bz2")
# BA = BALNLPModel("Dubrovnik/problem-273-176305-pre.txt.bz2")
# BA = BALNLPModel("Dubrovnik/problem-356-226730-pre.txt.bz2")
# BA = BALNLPModel("Venice/problem-427-310384-pre.txt.bz2")
# BA = BALNLPModel("Venice/problem-1350-894716-pre.txt.bz2")
# BA = BALNLPModel("Venice/problem-1778-993923-pre.txt.bz2")
# BA = BALNLPModel("Final/problem-4585-1324582-pre.txt.bz2")
# BA = BALNLPModel("Final/problem-13682-4456117-pre.txt.bz2")


# Wrap it into a NLS model
fr_BA = FeasibilityResidual(BA)

# Solve this problem using Levenberg-Marquardt algorithm
stats = Levenberg_Marquardt(fr_BA, :QR, :AMD)


# Write the output into a file

# using Logging
# io = open("ldl_amd_138.log", "w")
# file_logger = Logging.ConsoleLogger(io)
# stats = with_logger(file_logger) do
#   Levenberg_Marquardt(fr_BA, :LDL, :AMD)
# end
# flush(io)
# close(io)




# Spy

# BA = BALNLPModel("LadyBug/problem-138-19878-pre.txt.bz2")
# model = FeasibilityResidual(BA)
# x = model.meta.x0
# λ = 1.5
# rows = Vector{Int}(undef, model.nls_meta.nnzj)
# cols = Vector{Int}(undef, model.nls_meta.nnzj)
# jac_structure_residual!(model, rows, cols)
# vals = jac_coord_residual(model, x)
# perm = :AMD

# A = sparse(vcat(rows,collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(cols, collect(1 : model.meta.nvar)), vcat(vals, fill(sqrt(λ), model.meta.nvar)), model.nls_meta.nequ + model.meta.nvar, model.meta.nvar)
# if perm == :AMD
#     QR_A = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_AMD)
# elseif perm == :Metis
#     QR_A = myqr(A, ordering=SuiteSparse.SPQR.ORDERING_METIS)
# end
# print("\nok\n")
# spy(A[1 : model.nls_meta.nequ, :])
# spy(QR_A.R)

# cols .+= model.nls_meta.nequ
# A = sparse(vcat(collect(1 : model.nls_meta.nequ), rows, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(collect(1 : model.nls_meta.nequ), cols, collect(model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar)), vcat(fill(1.0, model.nls_meta.nequ), vals, fill(-λ, model.meta.nvar)))
# # col_norms = Vector{T}(undef, model.meta.nvar + model.nls_meta.nequ)
# if perm == :AMD
#     P = amd(A)
# elseif perm == :Metis
#     P , _ = Metis.permutation(A' + A)
#     P = convert(Array{Int64,1}, P)
# end
# ldl_symbolic = ldl_analyse(A, P, upper=true, n=model.meta.nvar + model.nls_meta.nequ)
# LDLT = ldl_factorize(A, ldl_symbolic, true)
# print("\nok\n")
# spy(A[1 : model.nls_meta.nequ, model.nls_meta.nequ + 1 : model.nls_meta.nequ + model.meta.nvar])
# spy(LDLT.L)



# # Test in other precisions

# using Quadmath
# BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# x0_simple = convert(Array{Float32,1}, fr_BA.meta.x0)
# # x0_quad = convert(Array{Float128,1}, fr_BA.meta.x0)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :Metis, x0_simple)
# print("\n", typeof(stats.solution), "\n")
# print("\n ------------ \nStats : \n", stats)
