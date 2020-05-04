using Quadmath
using SolverBenchmark
using SolverTools
using DataFrames
include("BALNLPModels.jl")
include("LevenbergMarquardt.jl")

# solvers = Dict(
#   :lmqramd => model -> Levenberg_Marquardt(model, :QR, :AMD),
#   :lmqrmetis => model -> Levenberg_Marquardt(model, :QR, :Metis),
#   :lmldlamd => model -> Levenberg_Marquardt(model,:LDL, :AMD),
#   :lmldlmetis => model -> Levenberg_Marquardt(model,:LDL, :Metis)
# )
#
# prob_names = ("LadyBug/problem-49-7776-pre.txt.bz2",
#               "LadyBug/problem-73-11032-pre.txt.bz2"
#               )
# problems = (FeasibilityResidual(BALNLPModel(name)) for name in prob_names)  # remarque les parenthèses
#
# stats = bmark_solvers(solvers, problems)
# print(markdown_table(stdout, stats[:lmqramd], cols=[:name, :f, :t]))
# df = join(stats, [:name, :nvar, :nequ, :status, :objective, :elapsed_time, :iter, :dual_feas])
# latex_table(stdout, stats[:lmqramd])


# # Create a BALNLPModel from a dataset
BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
# Wrap it into a NLS model
fr_BA = FeasibilityResidual(BA)
# Solve this problem using Levenberg-Marquardt algorithm
stats = Levenberg_Marquardt(fr_BA, :QR, :Metis)
print("\n ------------ \nStats : \n", stats)

BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
fr_BA = FeasibilityResidual(BA)
stats = Levenberg_Marquardt(fr_BA, :LDL, :Metis)
print("\n ------------ \nStats : \n", stats)


# # Test in other precisions

# BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# x0_simple = convert(Array{Float32,1}, fr_BA.meta.x0)
# # x0_quad = convert(Array{Float128,1}, fr_BA.meta.x0)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :Metis, x0_simple)
# print("\n", typeof(stats.solution), "\n")
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("LadyBug/problem-73-11032-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, :AMD)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("LadyBug/problem-73-11032-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("LadyBug/problem-138-19878-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, :AMD)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("LadyBug/problem-138-19878-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("LadyBug/problem-318-41628-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("LadyBug/problem-318-41628-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("LadyBug/problem-1723-156502-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)





# # Dubrovnik


# BA = BALNLPModel("Dubrovnik/problem-16-22106-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Dubrovnik/problem-16-22106-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("Dubrovnik/problem-88-64298-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Dubrovnik/problem-135-90642-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Dubrovnik/problem-173-111908-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Dubrovnik/problem-356-226730-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)




# # Trafalgar


# BA = BALNLPModel("Trafalgar/problem-21-11315-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Trafalgar/problem-21-11315-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("Trafalgar/problem-39-18060-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Trafalgar/problem-39-18060-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("Trafalgar/problem-257-65132-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Trafalgar/problem-257-65132-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)




# # Venice


# BA = BALNLPModel("Venice/problem-52-64053-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Venice/problem-427-310384-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)



# # Final


# BA = BALNLPModel("Final/problem-93-61203-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, :AMD, fr_BA.meta.x0)
# print("\n ------------ \nStats : \n", stats)
