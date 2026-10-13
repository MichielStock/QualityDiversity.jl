@testset "VariationEmitter" begin
    emitter = VariationEmitter(GaussianMutation(0.1); batch_size=7)
    @test QD.batch_size(emitter) == 7
    @test_throws "batch_size must be at least 1, got 0" VariationEmitter(GaussianMutation(0.1); batch_size=0)

    archive = GridArchive((4,), ((0, 1),))
    @test_throws "empty archive" QD.ask(Xoshiro(1), emitter, archive)
    QD.add!(archive, [0.0, 0.0], Evaluation(0.0, SVector(0.1)))
    xs = QD.ask(Xoshiro(1), emitter, archive)
    @test xs isa Vector{Vector{Float64}}
    @test length(xs) == 7
    @test all(x -> x != [0.0, 0.0], xs)
    @test QD.ask(Xoshiro(1), emitter, archive) == xs
end

@testset "VariationEmitter with a crossover" begin
    # An archive with two elites, [0,0,0] and [1,1,1]. Without isotropic noise a
    # child lies on the line through its two parents, so all entries are equal.
    archive = GridArchive((4,), ((0, 1),))
    QD.add!(archive, [0.0, 0.0, 0.0], Evaluation(0.0, SVector(0.1)))
    QD.add!(archive, [1.0, 1.0, 1.0], Evaluation(0.0, SVector(0.9)))

    emitter = VariationEmitter(IsoLineVariation(0.0, 1.0); batch_size=400)
    xs = QD.ask(Xoshiro(1), emitter, archive)
    @test xs isa Vector{Vector{Float64}}
    @test length(xs) == 400
    @test all(x -> x[1] == x[2] == x[3], xs)
    # parents differ in about half of the pairs, so many children leave {0, 1}
    @test count(x -> x[1] != 0 && x[1] != 1, xs) > 100
    @test QD.ask(Xoshiro(1), emitter, archive) == xs

    # with no noise at all every child is a copy of its first parent
    copies = QD.ask(Xoshiro(1), VariationEmitter(IsoLineVariation(0.0, 0.0); batch_size=50), archive)
    @test all(x -> x == [0.0, 0.0, 0.0] || x == [1.0, 1.0, 1.0], copies)
    @test any(==([0.0, 0.0, 0.0]), copies) && any(==([1.0, 1.0, 1.0]), copies)

    # a single elite is both parents
    single = GridArchive((4,), ((0, 1),))
    QD.add!(single, [2.0, 3.0], Evaluation(0.0, SVector(0.1)))
    @test all(==([2.0, 3.0]), QD.ask(Xoshiro(1), VariationEmitter(IsoLineVariation(0.0, 5.0); batch_size=10), single))
end
