using QualityDiversity.Testing

# A test set that counts failures and errors instead of reporting them, to check
# that the conformance checks reject types that break their contract.
mutable struct CountFailures <: Test.AbstractTestSet
    description::String
    failures::Int
end
CountFailures(description::AbstractString; kwargs...) = CountFailures(description, 0)
Test.record(ts::CountFailures, ::Union{Test.Fail,Test.Error}) = (ts.failures += 1; nothing)
Test.record(ts::CountFailures, ::Test.Pass) = nothing
Test.record(ts::CountFailures, ::Test.Broken) = nothing
Test.record(ts::CountFailures, child::CountFailures) = (ts.failures += child.failures; nothing)
function Test.finish(ts::CountFailures)
    Test.get_testset_depth() == 0 || Test.record(Test.get_testset(), ts)
    return ts
end

failures(f) = (@testset CountFailures "detect" begin f() end).failures

@testset "problems" begin
    test_problem(Rastrigin(2))
    test_problem(Rastrigin(7))
    test_problem(QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                           init=rng -> 2 .* rand(rng, 3) .- 1, bounds=((-3, 3), (-3, 3))))
end

@testset "archives" begin
    prob = Rastrigin(4)
    test_archive(() -> GridArchive(prob; dims=(8, 8)), prob)
    test_archive(() -> GridArchive(prob; dims=(1, 1)), prob)
    test_archive(() -> GridArchive(prob; dims=(5, 3)), prob)
    test_archive(() -> GridArchive(prob; dims=(8, 8), genotype=Vector{Float32}), prob)
    test_archive(() -> GridArchive(prob; dims=(8, 8), fitness=Float32), prob)
    test_archive(() -> GridArchive(prob; dims=(8, 8), outside=ClampOutside()), prob)
    test_archive(() -> GridArchive(prob; dims=(8, 8), outside=RejectOutside()), prob)
end

@testset "archive with an isbits genotype" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    test_archive(() -> GridArchive{SVector{3,Float64}}((6, 6), ((-1, 1), (-1, 1))), prob)
end

@testset "variation" begin
    test_variation(GaussianMutation(0.1), [0.5, -1.0, 2.0])
    test_variation(GaussianMutation(2), fill(0.0, 10))
    test_variation(IsoLineVariation(0.1, 0.5), [0.5, -1.0, 2.0]; parent2=[1.0, 0.0, -2.0])
    test_variation(IsoLineVariation(), fill(0.0, 10); parent2=fill(1.0, 10))
end

@testset "emitters" begin
    prob = Rastrigin(4)
    archive = GridArchive(prob; dims=(8, 8))
    xs = [QD.random_solution(Xoshiro(i), prob) for i in 1:20]
    QD.add_batch!(archive, xs, QD.evaluate_batch(prob, xs))
    test_emitter(VariationEmitter(GaussianMutation(0.2); batch_size=16), archive, prob)
    test_emitter(VariationEmitter(IsoLineVariation(0.05, 0.3); batch_size=16), archive, prob)
end

struct BadProblem <: AbstractQDProblem end
QD.evaluate(::BadProblem, x) = Evaluation(1.0, SVector(7.0, 0.0))   # outside the bounds
QD.random_solution(rng, ::BadProblem) = [1.0]
QD.descriptor_bounds(::BadProblem) = ((0.0, 1.0), (0.0, 1.0))

struct MutatingVariation <: AbstractVariation end
QD.vary(rng, ::MutatingVariation, parent) = (parent[1] += 1; parent)

@testset "the checks reject types that break their contract" begin
    @test failures(() -> test_problem(BadProblem())) > 0
    @test failures(() -> test_variation(MutatingVariation(), [1.0, 2.0])) > 0
    # a crossover that reports one parent, and a mutation checked as a crossover
    @test failures(() -> test_variation(IsoLineVariation(), [1.0, 2.0])) > 0
    @test failures(() -> test_variation(GaussianMutation(0.1), [1.0, 2.0]; parent2=[3.0, 4.0])) > 0
    @test failures(() -> test_problem(Rastrigin(4))) == 0
end
