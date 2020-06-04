using Test


# Test JuMP modeling
include("../src/ModelJuMP.jl")
@test Rodrigues_rotation([1.0, 1.0, 1.0], [2.5, -0.3, 1.0]) == [1.577353756980212, 2.1408840848258484, -0.5182378418060594]
@test scaling_factor([1.0 1.0], 1.0, 1.0) == 7.0
@test projection(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0) == [-7.0 -7.0]



# Test NLPmodels modeling
include("../src/BALNLPModels.jl")

pt2d = [-3.326500e+02, 2.620900e+02, -1.997600e+02, 1.667000e+02, -2.530600e+02, 2.022700e+02, 5.813000e+01, 2.718900e+02, 2.382200e+02, 2.373700e+02]
cam_idx = [1 2 3 4 5]
pnt_idx = [1 1 1 1 1]
x = [-0.6120001571722636, 0.5717590477602829, -1.8470812764548823, 0.01574151594294026, -0.012790936163850642, -0.004400849808198079, -0.034093839577186584, -0.10751387104921525, 1.1202240291236032, -3.177064385280358e-7, 5.882049053459402e-13, 399.75152639358436, 0.01597732412020533, -0.02522446458285646, -0.00940014164793023, -0.00856676614082241, -0.12188049069425422, 0.719013307500946, -3.7804765613385677e-7, 9.30743116838448e-13, 402.0175338595593, 0.014846251175275622, -0.021062899405576294, -0.0011669480098224182, -0.024950970734443037, -0.11398470545726247, 0.9216602073702798, -3.2952646187978145e-7, 6.732885068879348e-13, 400.4017536835857, 0.01991666998444233, -1.2243308199651954, 0.011998875602428538, -1.411897512312013, -0.11480651507716103, 0.44915582738113896, 5.958750036132224e-8, -2.4839062920074967e-13, 407.0302456821108, 0.02082242153136291, -1.238434791463721, 0.013893147632321344, -1.0496862247709429, -0.12995132856190453, 0.3379838023131856, 4.5673126640998776e-8, -1.7924276184384984e-13, 405.9176496201471]
nobs = 5
npnts = 1

r = zeros(Float64, 10)
residuals!(cam_idx, pnt_idx, x, r, nobs, npnts)
r .-= pt2d

true_residuals = [-9.020226301243156, 11.263958304987227, -1.833229714946924, 5.304698960898122, -4.332321480806684, 7.117305031392988, -0.5632751791502884, -1.062178017695942, -3.96920595468427, -2.285071283095334]
@test norm(true_residuals - r) == 0



# Test normalization
include("../src/lma_aux.jl")

# Create a sparse matrix A with a bloc √I at the bottom
m = 7
n = 5
λ = 1.5
rows = vcat(rand(1:m, 8), collect(m + 1 : m + n))
cols = vcat(rand(1:n, 8), collect(1 : n))
vals = vcat(rand(-4.5:4.5, 8), fill(sqrt(λ), n))
A = sparse(rows, cols, vals, m + n, n)
copy_A = copy(A)

# Test normalization of A for QR vesion
col_norms = Vector{eltype(A)}(undef, n)
normalize_qr_a!(A, col_norms, n)
for j = 1 : n
  @test abs(norm(A[:, j]) - 1) < 1e-10
  @test abs(col_norms[j] - norm(copy_A[:, j])) < 1e-10
end
denormalize_qr!(A, col_norms, n)
@test norm(A - copy_A) < 1e-10

# Test normalization of J for QR vesion
col_norms = Vector{eltype(A)}(undef, n)
normalize_qr_j!(A, col_norms, n)
for j = 1 : n
  if col_norms[j] != 0
    @test abs(norm(A[1 : m, j]) - 1) < 1e-10
    @test A[m + j, j] - copy_A[m + j, j] / col_norms[j] < 1e-10
  end
  @test abs(col_norms[j] - norm(copy_A[1 : m, j])) < 1e-10
end
denormalize_qr!(A, col_norms, n)
@test norm(A[1: m, :] - copy_A[1: m, :]) < 1e-10

# Create a sparse matrix A = [ [I  A₁₂]; [A₁₂ᵀ  -λI]]
m = 7
n = 5
λ = 1.5
rows = vcat(collect(1 : m), rand(1:m, 8), collect(m + 1 : m + n))
cols = vcat(collect(1 : m), rand(1:n, 8), collect(m + 1 : m + n))
vals = vcat(fill(1, m), rand(-4.5:4.5, 8), fill(-λ, n))
A = sparse(rows, cols, vals, m + n, m + n)
copy_A = copy(A)

# Test normalization of J for LDL vesion
col_norms = Vector{eltype(A)}(undef, n)
normalize_ldl!(A, col_norms, n, m)
for j = 1 : n
  if col_norms[j] != 0
    @test abs(norm(A[1 : m, m + j]) - 1) < 1e-10
    @test A[m + j, m + j] - copy_A[m + j, m + j] / col_norms[j]^2 < 1e-10
  end
  @test abs(col_norms[j] - norm(copy_A[1 : m, m + 1 : m + j])) < 1e-10
end
denormalize_ldl!(A, col_norms, n, m)
@test norm(A[1 : m, m + 1 : m + n] - copy_A[1 : m, m + 1 : m + n]) < 1e-10


# Test mul_sparse functions
m = 7
n = 5
nnz_a = 8
rows = rand(1:m, nnz_a)
cols = rand(1:n, nnz_a)
vals = rand(-4.5:4.5, nnz_a)
A = sparse(rows, cols, vals, m, n)
x1 = rand(-4.5:4.5, n)
x2 = rand(-4.5:4.5, n)

res = A * x1
xr = mul_sparse(rows, cols, vals, x1, nnz_a, m)
@test norm(xr -res) < 1e-10

res2 = A * x2
mul_sparse!(xr, rows, cols, vals, x2, nnz_a)
@test norm(xr -res2) < 1e-10


# Test my_qr and solve_qr!
include("../src/qr_aux.jl")
# Create a sparse matrix A with a bloc √I at the bottom
m = 7
n = 5
λ = 1.5
rows = vcat(rand(1:m, 8), collect(m + 1 : m + n))
cols = vcat(rand(1:n, 8), collect(1 : n))
vals = vcat(rand(-4.5:4.5, 8), fill(sqrt(λ), n))
A = sparse(rows, cols, vals, m + n, n)
b = rand(-4.5:4.5, m + n)

xr = similar(b)
QR_A = myqr(A)
x = solve_qr!(m + n, n, xr, b, QR_A.Q, QR_A.R, QR_A.prow, QR_A.pcol)

true_x = A \ b
@test norm(x - true_x) < 1e-10


# Test Givens strategy
m = 7
n = 5
λ = 1.5
rows = vcat([1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7], collect(m + 1 : m + n))
cols = vcat([1, 3, 2, 4, 4, 1, 4, 3, 1, 4, 2, 5], collect(1 : n))
vals = vcat([3.5, 2.3, -0.5, -1.5, 3.5, -4.5, -1.3, 2.7, 3.5, -4.5, -1.3, 2.7], fill(sqrt(λ), n))
A = sparse(rows, cols, vals, m + n, n)
b = rand(-4.5:4.5, m + n)


# QR facto of A
QR_A = qr(A[1 : m, :])
R = similar(QR_A.R)
nnz_R = nnz(QR_A.R)
G_list = Vector{LinearAlgebra.Givens{Float64}}(undef, Int(n*(n + 1)/2))
news = zeros(Float64, n)
Prow = vcat(QR_A.prow, collect(m + 1 : m + n))

# A_R is [R; √λI]
A_R = zeros(m + n, n)
A_R[1:n, 1:n] .= QR_A.R
for k = 1:n
  A_R[m+k, k] = sqrt(λ)
end

# Perform Givens rotations on QR_A.R
R .= QR_A.R
Rt = sparse(R')
counter = fullQR_givens!(R, Rt, G_list, news, sqrt(λ), n, m, nnz_R)

# Performs the same Givens rotations on A_R
A_R2 = similar(A_R)
for k = 1 : counter
	A_R2 = G_list[k] * A_R
	A_R .= A_R2
end

# Check if the √λ have been eliminated in A_R and if A_R = Rλ
@test norm(A_R[n+1:n+m, :]) < 1e-10
@test norm(A_R[1:n, :] - R) < 1e-10

# Check if Qλt_mul! works well
my_xr = similar(b)
Qλt_mul!(my_xr, QR_A.Q, G_list, b, n, m, counter)
true_xr = similar(b)
Qλ = Qλt_mul_verif!(true_xr, QR_A.Q, G_list, b, n, m, counter)
@test norm(my_xr - true_xr) < 1e-10

# Solve min ||[A; √λ] x - b|| with Givens strategy
xr = similar(b)
δ1 = solve_qr2!(m + n, n, xr, b, QR_A.Q, R, Prow, QR_A.pcol, counter, G_list)

# Check the results
true_delta = A \ b
@test norm(δ1 - true_delta) < 1e-10
