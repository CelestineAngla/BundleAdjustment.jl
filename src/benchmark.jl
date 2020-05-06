using SolverBenchmark
using SolverTools
using DataFrames
include("BALNLPModels.jl")
include("LevenbergMarquardt.jl")

# solvers = Dict(
#   :lmqramd => model -> Levenberg_Marquardt(model, :QR, :AMD),
#   :lmqrmetis => model -> Levenberg_Marquardt(model, :QR, :Metis),
#   :lmldlamd => model -> Levenberg_Marquardt(model,:LDL, :AMD),
#   :lmldlmetis => model -> Levenberg_Marquardt(model,:LDL, :Metis)
# )
#
# prob_names = ("LadyBug/problem-49-7776-pre.txt.bz2",
#               "LadyBug/problem-73-11032-pre.txt.bz2"
#               )
# problems = (FeasibilityResidual(BALNLPModel(name)) for name in prob_names)  # remarque les parenthèses
#
# stats = bmark_solvers(solvers, problems)
# df = join(stats, [:name, :nvar, :nequ, :status, :objective, :elapsed_time, :iter, :dual_feas])
# latex_table(stdout, df)
