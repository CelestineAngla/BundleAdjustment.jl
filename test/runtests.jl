using Test
include("../src/ReadFiles.jl")
include("../src/Model.jl")

@test Rodrigues_rotation([1.0, 1.0, 1.0], [2.5, -0.3, 1.0]) == [1.577353756980212, 2.1408840848258484, -0.5182378418060594]
@test scaling_factor([1.0 1.0], 1.0, 1.0) == 7.0
@test projection(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0) == [-7.0 -7.0]
