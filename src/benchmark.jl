using SolverBenchmark
using SolverTools
using DataFrames
include("BALNLPModels.jl")
include("LevenbergMarquardt.jl")

solvers = Dict(
  :lmqramd => model -> Levenberg_Marquardt(model, :QR, :AMD),
  :lmqrmetis => model -> Levenberg_Marquardt(model, :QR, :Metis),
  :lmldlamd => model -> Levenberg_Marquardt(model,:LDL, :AMD),
  :lmldlmetis => model -> Levenberg_Marquardt(model,:LDL, :Metis)
)

prob_names = ("LadyBug/problem-49-7776-pre.txt.bz2",
              "LadyBug/problem-73-11032-pre.txt.bz2",
              "LadyBug/problem-138-19878-pre.txt.bz2",
              "LadyBug/problem-318-41628-pre.txt.bz2",
              "LadyBug/problem-372-47423-pre.txt.bz2"
              )
problems = (FeasibilityResidual(BALNLPModel(name)) for name in prob_names)  # remarque les parenthèses

stats = bmark_solvers(solvers, problems)
save_stats(stats, "lm_stats.csv")
# df = join(stats, [:name, :nvar, :nequ, :status, :objective, :elapsed_time, :iter, :dual_feas])
# latex_table(stdout, df)
# markdown_table(stdout, df)

stats = load_stats("stats.csv")
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
