using Test
include("../src/ReadFiles.jl")
include("../src/Model.jl")

@test readfile("LadyBug/problem-49-7776-pre.txt") == readfile("LadyBug/problem-49-7776-pre.txt")

@test get_rotation_matrix([1.0 1.0 1.0]) == [[0.22629564095020646 -0.18300791965761704 0.9567122787074109]; [0.9567122787074109 0.22629564095020646 -0.18300791965761704]; [-0.18300791965761704 0.9567122787074109 0.22629564095020646]]
@test scaling_factor([1.0 1.0], 1.0, 1.0) == 7.0
@test projection(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0) == [-7.0000000000000036 -7.000000000000002]
