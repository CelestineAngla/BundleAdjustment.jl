using Logging
include("BALNLPModels.jl")
include("lm.jl")


prob_names = ("LadyBug/problem-49-7776-pre.txt.bz2",
              "LadyBug/problem-73-11032-pre.txt.bz2",
              "LadyBug/problem-138-19878-pre.txt.bz2",
              "LadyBug/problem-318-41628-pre.txt.bz2",
              "LadyBug/problem-460-56811-pre.txt.bz2",
              # "LadyBug/problem-646-73584-pre.txt.bz2",
              "LadyBug/problem-810-88814-pre.txt.bz2",
              # "LadyBug/problem-1031-110968-pre.txt.bz2",
              # "LadyBug/problem-1235-129634-pre.txt.bz2",
              "Dubrovnik/problem-202-132796-pre.txt.bz2",
              # "LadyBug/problem-1723-156502-pre.txt.bz2",
              # "Dubrovnik/problem-273-176305-pre.txt.bz2",
              # "Dubrovnik/problem-356-226730-pre.txt.bz2",
              # "Venice/problem-427-310384-pre.txt.bz2",
              # "Venice/problem-1350-894716-pre.txt.bz2"
              )


# # Float64
# f = open("lm_diffprec_F64.log", "w")
#
# for name in prob_names
#   problem = FeasibilityResidual(BALNLPModel(name))
#   logger = Logging.ConsoleLogger(f)
#   with_logger(logger) do
#     stats = Levenberg_Marquardt(problem, :LDL, :Metis, :None, false)
#     println(f, "\n\n", name, " & ", problem.meta.nvar, " & ", problem.nls_meta.nequ, " & ", stats.objective, " & ", stats.iter, " & ", stats.elapsed_time, " & ", stats.status, " & ", stats.dual_feas, " \\\\", "\n\n")
#   end
#   flush(f)
# end
# close(f)

# Float16/32 then Float64
f = open("lm_diffprec_F1632_64_2.log", "w")

for name in prob_names[7 : end]
  problem = FeasibilityResidual(BALNLPModel(name, Float32))
  logger = Logging.ConsoleLogger(f)
  x0_double = similar(problem.meta.x0, Float64)
  with_logger(logger) do
    stats1 = Levenberg_Marquardt(problem, :LDL, :Metis, :None, false, oatol=1e-2, ortol=1e-2, atol=1e-2, rtol=1e-1, satol=1e-5, srtol=1e-5, restol=1e-5, facto_type=Float16)
    println(f, "\n\n", name, " & ", problem.meta.nvar, " & ", problem.nls_meta.nequ, " & ", stats1.objective, " & ", stats1.iter, " & ", stats1.elapsed_time, " & ", stats1.status, " & ", stats1.dual_feas, " \\\\", "\n\n")
    x0_double = convert(Array{Float64,1}, stats1.solution)
  end
  flush(f)
  with_logger(logger) do
    stats2 = Levenberg_Marquardt(problem, :LDL, :Metis, :None, false; x=x0_double)
    println(f, "\n\n", name, " & ", problem.meta.nvar, " & ", problem.nls_meta.nequ, " & ", stats2.objective, " & ", stats2.iter, " & ", stats2.elapsed_time, " & ", stats2.status, " & ", stats2.dual_feas, " \\\\", "\n\n")
  end
  flush(f)
end
close(f)


# Float32/64
f = open("lm_diffprec_F3264.log", "w")

for name in prob_names
  problem = FeasibilityResidual(BALNLPModel(name))
  logger = Logging.ConsoleLogger(f)
  with_logger(logger) do
    stats = Levenberg_Marquardt(problem, :LDL, :Metis, :None, false, facto_type=Float32)
    println(f, "\n\n", name, " & ", problem.meta.nvar, " & ", problem.nls_meta.nequ, " & ", stats.objective, " & ", stats.iter, " & ", stats.elapsed_time, " & ", stats.status, " & ", stats.dual_feas, " \\\\", "\n\n")
  end
  flush(f)
end
close(f)


# Float16/32 then Float32/64
f = open("lm_diffprec_F1632_3264.log", "w")

for name in prob_names
  problem = FeasibilityResidual(BALNLPModel(name, Float32))
  logger = Logging.ConsoleLogger(f)
  x0_double = similar(problem.meta.x0, Float64)
  with_logger(logger) do
    stats1 = Levenberg_Marquardt(problem, :LDL, :Metis, :None, false, oatol=1e-2, ortol=1e-2, atol=1e-2, rtol=1e-1, satol=1e-5, srtol=1e-5, restol=1e-5, facto_type=Float16)
    println(f, "\n\n", name, " & ", problem.meta.nvar, " & ", problem.nls_meta.nequ, " & ", stats1.objective, " & ", stats1.iter, " & ", stats1.elapsed_time, " & ", stats1.status, " & ", stats1.dual_feas, " \\\\", "\n\n")
    x0_double = convert(Array{Float64,1}, stats1.solution)
  end
  flush(f)
  with_logger(logger) do
    stats2 = Levenberg_Marquardt(problem, :LDL, :Metis, :None, false; x=x0_double, facto_type=Float32)
    println(f, "\n\n", name, " & ", problem.meta.nvar, " & ", problem.nls_meta.nequ, " & ", stats2.objective, " & ", stats2.iter, " & ", stats2.elapsed_time, " & ", stats2.status, " & ", stats2.dual_feas, " \\\\", "\n\n")
  end
  flush(f)
end
close(f)
