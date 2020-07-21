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


# BA = BALNLPModel(prob_names[2], Float32)
# fr_BA = FeasibilityResidual(BA)
# x0_simple = convert(Array{Float32,1}, fr_BA.meta.x0)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_simple, oatol=1e-4, ortol=1e-4, atol=1e-4, rtol=1e-5, satol=1e-6, srtol=1e-7)
# x0_double = convert(Array{Float64,1}, stats.solution)
# BA.ignored_res = Int[]
# fr_BA = FeasibilityResidual(BA)
# stats2 = Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_double)

BA = BALNLPModel(prob_names[3], Float32)
fr_BA = FeasibilityResidual(BA)
stats1 = Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=fr_BA.meta.x0, oatol=1e-2, ortol=1e-2, atol=1e-2, rtol=1e-1, satol=1e-5, srtol=1e-6, restol=1e-5, facto_type=Float16, λ=1)
x0_double = convert(Array{Float64,1}, stats1.solution)
fr_BA = FeasibilityResidual(BA)
stats2 = Levenberg_Marquardt(fr_BA, :LDL, :AMD, :None, false; x=x0_double)

# 126661:126669
# 126661 10835 -56.019512
# 126662 10836 129.02536
# 126663 10837 -68.94727
# 126664 10838 156.85902
# 126665 10839 -46.270138
# 126666 10840 107.28246
# 126667 10841 -86.51637
# 126668 10842 192.91069
# 126669 94143 -293.12912
# 125988 125997
# -293.12912 37214.535
# -37507.664 7485.0825
# -44992.746 11509.527
# -56502.273 2140.9255
# -58643.2 24604.754
# -83247.95 4753.7266
# -88001.68 16647.543
# -104649.22 3138.1858
# -107787.41 -17803.83
# -89983.58 -88241.8

# SparseMatrixCSC{Float32,Int64}
# 126661:126669
# 126661 10835 -56.0
# 126662 10836 129.0
# 126663 10837 -69.0
# 126664 10838 157.0
# 126665 10839 -46.25
# 126666 10840 107.5
# 126667 10841 -86.5
# 126668 10842 193.0
# 126669 94143 -294.0
# 125988 125997
# BFloat16(-294.0) BFloat16(37376.0)
# BFloat16(-37632.0) BFloat16(7488.0)
# BFloat16(-45056.0) BFloat16(11584.0)
# BFloat16(-56576.0) BFloat16(2144.0)
# BFloat16(-58624.0) BFloat16(24704.0)
# BFloat16(-83456.0) BFloat16(4768.0)
# BFloat16(-88064.0) BFloat16(16640.0)
# BFloat16(-104448.0) BFloat16(3136.0)
# BFloat16(-107520.0) BFloat16(-17920.0)
# BFloat16(-89600.0) BFloat16(-89600.0)
