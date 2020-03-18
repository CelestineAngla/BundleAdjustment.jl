using CUTEst
fetch_sif_problems()
model = CUTEstModel("BA-L1LS")
# model = CUTEstModel("BA-L49LS")

# using NLPModelsIpopt
# stats = ipopt(model)

print("x0 = $( model.meta.x0 )")
print("fx = $( obj(model, model.meta.x0) )")
print("gx = $( grad(model, model.meta.x0) )")
print("Hx = $( hess(model, model.meta.x0) )")
