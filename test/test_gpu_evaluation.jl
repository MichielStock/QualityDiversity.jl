# Evaluating a batch on an array type other than Array. JLArrays runs the array
# operations on the CPU, so this checks the logic of the example without a GPU.
using JLArrays: JLArray
using QualityDiversity.Testing

include(joinpath(@__DIR__, "..", "examples", "gpu_evaluation.jl"))
using .GPUExample: GPURastrigin, backend

@testset "evaluate_batch on device arrays" begin
    prob = GPURastrigin(JLArray, 20)
    cpu = Rastrigin(20)
    rng = Xoshiro(1)
    xs = [QD.random_solution(rng, prob) for _ in 1:200]

    evs = QD.evaluate_batch(prob, xs)
    @test evs isa Vector{Evaluation{Float32,SVector{2,Float64}}}
    @test length(evs) == 200
    # the Float64 CPU benchmark agrees up to Float32 rounding
    for (x, ev) in zip(xs, evs)
        ref = QD.evaluate(cpu, Float64.(x))
        @test ev.fitness ≈ ref.fitness rtol = 1e-4
        @test ev.descriptor ≈ ref.descriptor atol = 1e-4
    end
    # the CPU evaluation of the same type agrees with the device batch
    @test all(isapprox(ev.fitness, QD.evaluate(prob, x).fitness; rtol=1e-5) for (x, ev) in zip(xs, evs))

    # sizes of the batch
    @test QD.evaluate_batch(prob, xs[1:1]) == evs[1:1]
    @test QD.evaluate_batch(prob, empty(xs)) == Evaluation{Float32,SVector{2,Float64}}[]
    # Float64 genotypes are accepted
    @test QD.evaluate_batch(prob, [Float64.(x) for x in xs[1:5]]) == evs[1:5]

    # solutions far outside the box are clipped, and the descriptor stays inside it
    extreme = [fill(1f6, 20), fill(-1f6, 20), fill(5.12f0, 20), fill(-5.12f0, 20)]
    for ev in QD.evaluate_batch(prob, extreme)
        @test all(abs.(ev.descriptor) .<= 5.12)
    end
    @test QD.evaluate_batch(prob, extreme[1:1])[1].descriptor ≈ SVector(5.12, 5.12) atol = 1e-5

    @test_throws "every solution must have 20 entries" QD.evaluate_batch(prob, [zeros(Float32, 20), zeros(Float32, 19)])
    @test_throws "solution has 3 entries but GPURastrigin has dimension 20" QD.evaluate(prob, zeros(3))
    @test_throws "GPURastrigin needs at least 2 dimensions, got 1" GPURastrigin(JLArray, 1)
end

# A problem whose batch evaluation is off by a relative `error`, as one that
# computes in lower precision on a device is.
struct SlightlyOff <: AbstractQDProblem
    error::Float64
end
QD.evaluate(::SlightlyOff, x) = QD.evaluate(Rastrigin(4), x)
QD.random_solution(rng, ::SlightlyOff) = QD.random_solution(rng, Rastrigin(4))
QD.descriptor_bounds(::SlightlyOff) = QD.descriptor_bounds(Rastrigin(4))
function QD.evaluate_batch(prob::SlightlyOff, xs::AbstractVector)
    return [Evaluation(ev.fitness * (1 + prob.error), ev.descriptor) for ev in map(x -> QD.evaluate(Rastrigin(4), x), xs)]
end

@testset "the conformance check accepts a tolerance" begin
    test_problem(GPURastrigin(JLArray, 6); rtol=1e-4)
    test_problem(GPURastrigin(JLArray, 7); rtol=1e-4)

    off = SlightlyOff(1e-6)
    test_problem(off; rtol=1e-4)
    @test failures(() -> test_problem(off)) > 0                  # exact comparison fails
    @test failures(() -> test_problem(off; rtol=1e-9)) > 0       # a tolerance below the error fails
    @test failures(() -> test_problem(SlightlyOff(0.1); rtol=1e-4)) > 0
    @test failures(() -> test_problem(SlightlyOff(0.0))) == 0
end

@testset "MAP-Elites with a device-evaluated problem" begin
    prob = GPURastrigin(JLArray, 6)
    alg = MAPElites(prob; dims=(10, 10), batch_size=200, genotype=Vector{Float32}, fitness=Float32,
                    variation=GaussianMutation(0.5f0))
    @test alg.archive isa GridArchive{Vector{Float32},Float32}
    result = solve(prob, alg; budget=6_000, rng=Xoshiro(1))
    @test result.n_evals == 6_000
    @test length(result.archive) > 20
    @test all(g isa Vector{Float32} for (g, _) in QD.elites(result.archive))
    @test solve(prob, alg; budget=6_000, rng=Xoshiro(1)).history == result.history
    # close to the CPU run in quality
    cpu = MAPElites(Rastrigin(6); dims=(10, 10), batch_size=200, variation=GaussianMutation(0.5))
    cpu_result = solve(Rastrigin(6), cpu; budget=6_000, rng=Xoshiro(1))
    @test abs(coverage(result.archive) - coverage(cpu_result.archive)) < 0.1
end

@testset "choosing a backend" begin
    @test backend("JLArray") === JLArray
    @test_throws "unknown backend Vulkan; use JLArray, CUDA, Metal, AMDGPU or oneAPI" backend("Vulkan")
    # none of the GPU packages is a dependency of the test environment
    @test_throws "backend CUDA is not installed" backend("CUDA")
end
