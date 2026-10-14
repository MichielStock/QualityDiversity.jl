using QualityDiversity
using StaticArrays: SVector
using Random: Xoshiro, randperm
using Test

const QD = QualityDiversity

@testset "QualityDiversity" begin
    @testset "contracts" begin include("test_contracts.jl") end
    @testset "problems" begin include("test_problems.jl") end
    @testset "archives" begin include("test_archives.jl") end
    @testset "cvt archive" begin include("test_cvt.jl") end
    @testset "categorical archive" begin include("test_categorical.jl") end
    @testset "unstructured archive" begin include("test_unstructured.jl") end
    @testset "variation" begin include("test_variation.jl") end
    @testset "emitters" begin include("test_emitters.jl") end
    @testset "solve" begin include("test_solve.jl") end
    @testset "conformance" begin include("test_conformance.jl") end
    @testset "graph example" begin include("test_graphs.jl") end
    @testset "gpu evaluation example" begin include("test_gpu_evaluation.jl") end
    @testset "inference and allocations" begin include("test_inference.jl") end
    @testset "quality" begin include("test_quality.jl") end
end
