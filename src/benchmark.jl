using SolverBenchmark
using SolverTools
include("BALNLPModels.jl")
include("lm.jl")

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
              "LadyBug/problem-646-73584-pre.txt.bz2",
              "LadyBug/problem-1031-110968-pre.txt.bz2",
              "LadyBug/problem-1723-156502-pre.txt.bz2",
              "Dubrovnik/problem-356-226730-pre.txt.bz2",
              "Venice/problem-1350-894716-pre.txt.bz2"
              # "Final/problem-4585-1324582-pre.txt.bz2"
              )
problems = (FeasibilityResidual(BALNLPModel(name)) for name in prob_names)

using Logging
io = open("lm.log", "w")
stats = bmark_solvers(solvers, problems, solver_logger = Logging.ConsoleLogger(io))
flush(io)
close(io)
save_stats(stats, "lm_stats.csv")

for solver in solvers
  open(String(solver.first) * "_table.log","w") do io
    latex_table(io, stats[solver.first], cols=[:name, :nvar, :nequ, :objective, :elapsed_time, :iter,  :status, :dual_feas])
    markdown_table(io, stats[solver.first], cols=[:name, :nvar, :nequ, :objective, :elapsed_time, :iter,  :status, :dual_feas])
  end
end

using Plots
gr()
ENV["GKSwstype"] = "100"
solved(stats) = stats.status .!= :neg_pred
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
