using SparsityDetection, SparseArrays, SparseDiffTools, ModelingToolkit
include("lma_aux.jl")
include("BALNLPModels.jl")

prec = Float64
# BA = BALNLPModel("LadyBug/problem-49-7776-pre.txt.bz2", prec)
BA = BALNLPModel("LadyBug/problem-73-11032-pre.txt.bz2", prec)
# BA = BALNLPModel("LadyBug/problem-138-19878-pre.txt.bz2", prec)
# BA = BALNLPModel("LadyBug/problem-318-41628-pre.txt.bz2", prec)
# BA = BALNLPModel("LadyBug/problem-460-56811-pre.txt.bz2", prec)
# BA = BALNLPModel("LadyBug/problem-646-73584-pre.txt.bz2", prec)
# BA = BALNLPModel("LadyBug/problem-810-88814-pre.txt.bz2", prec)
# BA = BALNLPModel("LadyBug/problem-1031-110968-pre.txt.bz2", prec)
# BA = BALNLPModel("LadyBug/problem-1235-129634-pre.txt.bz2", prec)
# BA = BALNLPModel("LadyBug/problem-1723-156502-pre.txt.bz2", prec)
# BA = BALNLPModel("Dubrovnik/problem-202-132796-pre.txt.bz2", prec)
# BA = BALNLPModel("Dubrovnik/problem-273-176305-pre.txt.bz2", prec)
# BA = BALNLPModel("Dubrovnik/problem-356-226730-pre.txt.bz2", prec)
# BA = BALNLPModel("Venice/problem-427-310384-pre.txt.bz2", prec)
# BA = BALNLPModel("Venice/problem-1350-894716-pre.txt.bz2", prec)
model = FeasibilityResidual(BA)

input1 = rand(model.meta.nvar)
output1 = similar(input1, model.nls_meta.nequ)
r = (output1, input1) -> residual!(model, input1, output1)

# rows = Vector{Int}(undef, model.nls_meta.nnzj)
# cols = Vector{Int}(undef, model.nls_meta.nnzj)
# jac_structure_residual!(model, rows, cols)

# x = rand(model.meta.nvar)
@variables x[1 : model.meta.nvar]
y = residual(model, x)
j_ad = ModelingToolkit.jacobian_sparsity(y, x)
print("\n", j_ad, "\n")

# Automatic differentiation
# j_ad = sparse(rows, cols, prec(1.0))
j_ad = prec.(sparse(sparsity_pattern))
sparsity_pattern = jacobian_sparsity(r, output1, input1)
colors = matrix_colors(j_ad)
@time begin
forwarddiff_color_jacobian!(j_ad, r, model.meta.x0, colorvec = colors)
end
print("\n", norm(j_ad))

# Jacobian by hand
vals = Vector{prec}(undef, model.nls_meta.nnzj)
@time begin
jac_coord_residual!(model, model.meta.x0, vals)
j = sparse(rows, cols, vals)
end
print("\n", norm(j))


print("\n", norm(j - j_ad) / norm(j))
