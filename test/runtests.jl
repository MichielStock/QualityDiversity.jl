using QualityDiversity
using StaticArrays: SVector
using Random: Xoshiro, randperm
using LinearAlgebra: I
using Test

const QD = QualityDiversity

@testset "QualityDiversity" begin
    @testset "contracts" begin include("test_contracts.jl") end
    @testset "problems" begin include("test_problems.jl") end
    @testset "stochastic problems" begin include("test_stochastic.jl") end
    @testset "pareto archive and MOME" begin include("test_pareto.jl") end
    @testset "archives" begin include("test_archives.jl") end
    @testset "sliding boundaries archive" begin include("test_sliding.jl") end
    @testset "cvt archive" begin include("test_cvt.jl") end
    @testset "categorical archive" begin include("test_categorical.jl") end
    @testset "threshold archive" begin include("test_threshold.jl") end
    @testset "spread archive and MELS" begin include("test_spread.jl") end
    @testset "unstructured archive" begin include("test_unstructured.jl") end
    @testset "dns archive" begin include("test_dns.jl") end
    @testset "density archive and DDS" begin include("test_density.jl") end
    @testset "variation" begin include("test_variation.jl") end
    @testset "emitters" begin include("test_emitters.jl") end
    @testset "cma-es" begin include("test_cma_es.jl") end
    @testset "evolution strategies" begin include("test_evolution_strategies.jl") end
    @testset "cma emitter" begin include("test_cma_emitter.jl") end
    @testset "bandit scheduler" begin include("test_bandit.jl") end
    @testset "solve" begin include("test_solve.jl") end
    @testset "cqd score" begin include("test_cqd.jl") end
    @testset "gradients and differentiable QD" begin include("test_gradients.jl") end
    @testset "conformance" begin include("test_conformance.jl") end
    @testset "phenotype" begin include("test_phenotype.jl") end
    @testset "graph example" begin include("test_graphs.jl") end
    @testset "gpu evaluation example" begin include("test_gpu_evaluation.jl") end
    @testset "device archive and emitter" begin include("test_device.jl") end
    @testset "makie extension" begin include("test_makie.jl") end
    @testset "inference and allocations" begin include("test_inference.jl") end
    @testset "quality" begin include("test_quality.jl") end
end
