using CUTEst
using NLPModels


# fetch_sif_problems()
# model = CUTEstModel("BA-L1LS")

# finalize(BA49)
BA49 = CUTEstModel("BA-L49")

# using NLPModelsIpopt
# stats = ipopt(model)

J = jac(BA49, BA49.meta.x0)

for j = 1 : 23769
    if J[1,j] != 0
        print("\n", j, " ", J[1,j])
        print("\n", j, " ", J[2,j])
    end
end

# err = jacobian_check(BA49, x=BA49.meta.x0)
# print(err)

# cx = Vector{Float64}(undef, BA49.meta.ncon)
# cx = cons!(BA49, BA49.meta.x0, cx)
# cx_sort = sort(cx)
# print(cx[1:100])
finalize(BA49)
