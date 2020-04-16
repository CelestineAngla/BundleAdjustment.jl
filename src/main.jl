include("BALNLPModels.jl")
include("LevenbergMarquardt.jl")


# Create a BALNLPModel from a dataset
BA49 = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")

# Wrap it into a NLS model
fr_BA49 = FeasibilityResidual(BA49)

# Solve this problem using Levenberg-Marquardt algorithm
stats = Levenberg_Marquardt(fr_BA49, :QR, fr_BA49.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
print("\n ------------ \nStats : \n", stats)

# stats = Levenberg_Marquardt(fr_BA49, :LDL, fr_BA49.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA138 = BALNLPModel("LadyBug/problem-138-19878-pre.txt.bz2")
# fr_BA138 = FeasibilityResidual(BA138)
# x_opt = @time Levenberg_Marquardt(fr_BA138, fr_BA138.meta.x0, 1e-1, 1e-3, 1e-5, 1e-3, 3.0, 1.5, 1.5, 100)

# BA318 = BALNLPModel("LadyBug/problem-318-41628-pre.txt.bz2")
# fr_BA318 = FeasibilityResidual(BA318)
# x_opt = @time Levenberg_Marquardt(fr_BA318, fr_BA318.meta.x0, 1e-1, 1e-3, 1e-5, 1e-3, 3.0, 1.5, 1.5, 100)


# BA13682 = BALNLPModel("Final/problem-13682-4456117-pre.txt.bz2")
# fr_BA13682 = FeasibilityResidual(BA13682)
# x_opt = @time Levenberg_Marquardt(fr_BA13682, fr_BA13682.meta.x0, 1e-1, 1e-4, 1e-5, 5e-3, 5.0, 1.5, 1.5, 100)
