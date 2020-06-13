include("BALNLPModels.jl")
include("lm.jl")


prob_names = ("LadyBug/problem-49-7776-pre.txt.bz2",
              "LadyBug/problem-73-11032-pre.txt.bz2",
              "LadyBug/problem-318-41628-pre.txt.bz2",
              "LadyBug/problem-460-56811-pre.txt.bz2",
              "LadyBug/problem-810-88814-pre.txt.bz2",
              "LadyBug/problem-1031-110968-pre.txt.bz2",
              "Dubrovnik/problem-202-132796-pre.txt.bz2",
              "Dubrovnik/problem-273-176305-pre.txt.bz2",
              "Dubrovnik/problem-356-226730-pre.txt.bz2",
              "Venice/problem-427-310384-pre.txt.bz2",
              "Venice/problem-1350-894716-pre.txt.bz2"
              )
using Logging
io = open("diffprec.log", "w")
file_logger = Logging.ConsoleLogger(io)
for pb in prob_names
  BA = BALNLPModel(pb)
  fr_BA = FeasibilityResidual(BA)
  x0_simple = convert(Array{Float32,1}, fr_BA.meta.x0)
  stats = with_logger(file_logger) do
    Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_simple, oatol=1e-4, ortol=1e-3, atol=1e-4, rtol=1e-3, satol=1e-4, srtol=1e-4)
  end
  x0_double = convert(Array{Float64,1}, stats.solution)
  stats2 = with_logger(file_logger) do
    Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_double)
  end
end


flush(io)
close(io)
