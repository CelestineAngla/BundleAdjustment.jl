using CUTEst
fetch_sif_problems()
# model = CUTEstModel("BA-L1LS")
model = CUTEstModel("BA-L49LS")

using NLPModelsIpopt
stats = ipopt(model)
