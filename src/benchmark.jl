using SolverBenchmark
using SolverTools
using DataFrames
include("BALNLPModels.jl")
include("LevenbergMarquardt.jl")

solvers = Dict(
  :lmqramdnone => model -> Levenberg_Marquardt(model, :QR, :AMD, :None),
  :lmqramdj => model -> Levenberg_Marquardt(model, :QR, :AMD, :J),
  :lmqramda => model -> Levenberg_Marquardt(model, :QR, :AMD, :A),
  :lmqrmetis => model -> Levenberg_Marquardt(model, :QR, :Metis, :None),
  :lmldlamdnone => model -> Levenberg_Marquardt(model,:LDL, :AMD, :None),
  :lmldlamdj => model -> Levenberg_Marquardt(model,:LDL, :AMD, :J),
  :lmldlmetis => model -> Levenberg_Marquardt(model,:LDL, :Metis, :None)
)

prob_names = ("LadyBug/problem-49-7776-pre.txt.bz2",
              "LadyBug/problem-73-11032-pre.txt.bz2",
              "LadyBug/problem-138-19878-pre.txt.bz2",
              "LadyBug/problem-318-41628-pre.txt.bz2"
              )
problems = (FeasibilityResidual(BALNLPModel(name)) for name in prob_names)  # remarque les parenthèses

using Logging
stats = bmark_solvers(solvers, problems, solver_logger = Logging.ConsoleLogger())
save_stats(stats, "lm_stats.csv")

# df = join(stats, [:name, :nvar, :nequ, :status, :objective, :elapsed_time, :iter, :dual_feas])
# latex_table(stdout, df)
# markdown_table(stdout, df)

# joinpath(@__DIR__, "..", "lm_stats.csv")
# stats = load_stats(joinpath(@__DIR__, "..", "lm_stats.csv"))
# print(latex_table(stdout, stats[:lmldlmetis], cols=[:nvar, :nequ, :status, :objective, :elapsed_time, :iter, :primal_feas]))

using Plots
gr()
ENV["GKSwstype"] = "100"
solved(stats) = stats.status .== stats.status
costnames = ["time",
             "r evals",
             "J evals",
            ]
costs = [stats -> .!solved(stats) .* Inf .+ stats.elapsed_time,
         stats -> .!solved(stats) .* Inf .+ stats.neval_residual,
         stats -> .!solved(stats) .* Inf .+ stats.neval_jac_residual,
        ]
profile_solvers(stats, costs, costnames)
savefig("lm_profiles.pdf")
