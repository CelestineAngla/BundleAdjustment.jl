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

if ARGS[4] == "None"
    norm = :None
elseif ARGS[4] == "A"
    norm = :A
elseif ARGS[4] == "J"
    norm = :J
end

BA = BALNLPModel(ARGS[1])
fr_BA = FeasibilityResidual(BA)
stats = Levenberg_Marquardt(fr_BA, facto, perm, norm)
print("\n ------------ \nStats : \n", stats)
