include("BALNLPModels.jl")
include("lm.jl")


prob_names = ("LadyBug/problem-73-11032-pre.txt.bz2",
              "LadyBug/problem-138-19878-pre.txt.bz2",
              "LadyBug/problem-318-41628-pre.txt.bz2",
              "LadyBug/problem-460-56811-pre.txt.bz2")


BA = BALNLPModel(prob_names[5], Float32)
fr_BA = FeasibilityResidual(BA)
x0_simple = convert(Array{Float32,1}, fr_BA.meta.x0)
stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_simple, oatol=1e-4, ortol=1e-4, atol=1e-4, rtol=1e-3, satol=1e-6, srtol=1e-7)
x0_double = convert(Array{Float64,1}, stats.solution)
stats2 = Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_double)
