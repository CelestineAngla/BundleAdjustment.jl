using SparsityDetection, SparseArrays, SparseDiffTools
include("BALNLPModels.jl")

BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2")
model = FeasibilityResidual(BA)
input = rand(model.meta.nvar)
output = similar(input, model.nls_meta.nequ)
r = (output, input) -> residual!(model, input, output)
rows = Vector{Int}(undef, model.nls_meta.nnzj)
cols = Vector{Int}(undef, model.nls_meta.nnzj)
jac_structure_residual!(model, rows, cols)
sparsity_pattern = Sparsity(model.nls_meta.nequ, model.meta.nvar, rows, cols)
# sparsity_pattern = jacobian_sparsity(r, output, input)
jac = Float64.(sparse(sparsity_pattern))  # replace Float64 with the precision being used
colors = matrix_colors(jac)
forwarddiff_color_jacobian!(jac, r, model.meta.x0, colorvec = colors)

vals = jac_coord_residual(model, model.meta.x0)
j = sparse(rows, cols, vals)
print("\n", norm(j - jac))
