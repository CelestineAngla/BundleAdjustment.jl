include("BALNLPModels.jl")
include("LevenbergMarquardt.jl")


# Create a BALNLPModel from a dataset
BA49 = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")

# Wrap it into a NLS model
fr_BA49 = FeasibilityResidual(BA49)

# Solve this problem using Levenberg-Marquardt algorithm
x_opt = @time Levenberg_Marquardt(fr_BA49, fr_BA49.meta.x0, 1e-5, 1e-2, 100000)
