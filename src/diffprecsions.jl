include("BALNLPModels.jl")
include("lm.jl")
using BFloat16s


prob_names = ("LadyBug/problem-49-7776-pre.txt.bz2",
                "LadyBug/problem-73-11032-pre.txt.bz2",
                "LadyBug/problem-138-19878-pre.txt.bz2",
                "LadyBug/problem-318-41628-pre.txt.bz2",
                "LadyBug/problem-460-56811-pre.txt.bz2",
                "LadyBug/problem-646-73584-pre.txt.bz2",
                "LadyBug/problem-810-88814-pre.txt.bz2",
                "LadyBug/problem-1031-110968-pre.txt.bz2",
                "Dubrovnik/problem-202-132796-pre.txt.bz2")


# Float32 then Float64

# BA = BALNLPModel(prob_names[2], Float32)
# fr_BA = FeasibilityResidual(BA)
# x0_simple = convert(Array{Float32,1}, fr_BA.meta.x0)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_simple, oatol=1e-4, ortol=1e-4, atol=1e-4, rtol=1e-5, satol=1e-6, srtol=1e-7)
# x0_double = convert(Array{Float64,1}, stats.solution)
# BA.ignored_res = Int[]
# fr_BA = FeasibilityResidual(BA)
# stats2 = Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_double)

# Float16/32 then Float64

# BA = BALNLPModel(prob_names[2], Float32)
# fr_BA = FeasibilityResidual(BA)
# stats1 = Levenberg_Marquardt(fr_BA, :LDL, :Metis, :None, false; x=fr_BA.meta.x0, oatol=1e-2, ortol=1e-2, atol=1e-2, rtol=1e-1, satol=1e-5, srtol=1e-5, restol=1e-5, facto_type=Float16, iterative_refinement=true)
# x0_double = convert(Array{Float64,1}, stats1.solution)
# fr_BA = FeasibilityResidual(BA)
# stats2 = Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_double)

# Float32/64

BA = BALNLPModel(prob_names[1])
fr_BA = FeasibilityResidual(BA)
stats = Levenberg_Marquardt(fr_BA, :LDL, :Metis, :None, false; facto_type=Float32)

# Float16/32 then Float32/64

# BA = BALNLPModel(prob_names[4], Float32)
# fr_BA = FeasibilityResidual(BA)
# stats1 = Levenberg_Marquardt(fr_BA, :LDL, :Metis, :None, false; x=fr_BA.meta.x0, oatol=1e-2, ortol=1e-2, atol=1e-2, rtol=1e-1, satol=1e-5, srtol=1e-5, restol=1e-5, facto_type=Float16)
# x0_double = convert(Array{Float64,1}, stats1.solution)
# fr_BA = FeasibilityResidual(BA)
# stats2 = Levenberg_Marquardt(fr_BA, :LDL, :Metis, :None, false; x=x0_double, facto_type=Float32)
