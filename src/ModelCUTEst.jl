using CUTEst
fetch_sif_problems()
# model = CUTEstModel("BA-L1LS")
finalize(BA49)
BA49 = CUTEstModel("BA-L49LS")

# using NLPModelsIpopt
# stats = ipopt(model)

print("x0 = $( BA49.meta.x0 )")
print("J = $( jac(BA49, BA49.meta.x0) )")
