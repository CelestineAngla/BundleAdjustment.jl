include("BALNLPModels.jl")
include("LevenbergMarquardt.jl")

if ARGS[2] == "QR"
    facto = :QR
elseif ARGS[2] == "LDL"
    facto = :LDL
end

if ARGS[3] == "AMD"
    perm = :AMD
elseif ARGS[3] == "Metis"
    perm = :Metis
end

BA = BALNLPModel(ARGS[1])
fr_BA = FeasibilityResidual(BA)
stats = Levenberg_Marquardt(fr_BA, facto, perm)
print("\n ------------ \nStats : \n", stats)
