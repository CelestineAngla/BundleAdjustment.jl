using CUTEst
using NLPModels


# fetch_sif_problems()
# model = CUTEstModel("BA-L1LS")
finalize(BA49)
BA49 = CUTEstModel("BA-L49")

# using NLPModelsIpopt
# stats = ipopt(model)

J = jac(BA49, BA49.meta.x0)
print(J)
finalize(BA49)
