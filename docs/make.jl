# push!(LOAD_PATH,"../src/")
using Documenter

include("../src/BundleAdjustment.jl")

using .BundleAdjustment

makedocs(modules = [BundleAdjustment], sitename = "BundleAdjustment.jl", format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"))

deploydocs(repo = "github.com/CelestineAngla/BundleAdjustment.jl.git")
