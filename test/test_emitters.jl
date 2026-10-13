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
