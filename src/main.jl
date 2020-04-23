include("BALNLPModels.jl")
include("LevenbergMarquardt.jl")


# # Create a BALNLPModel from a dataset
# BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
# # Wrap it into a NLS model
# fr_BA = FeasibilityResidual(BA)
# # Solve this problem using Levenberg-Marquardt algorithm
# stats = Levenberg_Marquardt(fr_BA, :QR, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("LadyBug/problem-73-11032-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("LadyBug/problem-73-11032-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("LadyBug/problem-138-19878-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("LadyBug/problem-138-19878-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("LadyBug/problem-318-41628-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("LadyBug/problem-318-41628-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("LadyBug/problem-1723-156502-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)





# # Dubrovnik


# BA = BALNLPModel("Dubrovnik/problem-16-22106-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Dubrovnik/problem-16-22106-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("Dubrovnik/problem-88-64298-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Dubrovnik/problem-135-90642-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Dubrovnik/problem-173-111908-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Dubrovnik/problem-356-226730-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)




# # Trafalgar


# BA = BALNLPModel("Trafalgar/problem-21-11315-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Trafalgar/problem-21-11315-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("Trafalgar/problem-39-18060-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Trafalgar/problem-39-18060-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)



# BA = BALNLPModel("Trafalgar/problem-257-65132-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :QR, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Trafalgar/problem-257-65132-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)




# # Venice


# BA = BALNLPModel("Venice/problem-52-64053-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)

# BA = BALNLPModel("Venice/problem-427-310384-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)



# # Final


# BA = BALNLPModel("Final/problem-93-61203-pre.txt.bz2")
# fr_BA = FeasibilityResidual(BA)
# stats = Levenberg_Marquardt(fr_BA, :LDL, fr_BA.meta.x0, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-2, 3.0, 3.0, 1.5, 100)
# print("\n ------------ \nStats : \n", stats)
