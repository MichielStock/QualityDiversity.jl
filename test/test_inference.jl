# Calls on the hot path infer a concrete type and, for isbits genotypes, do not
# allocate. Measurements are taken inside functions after a warm-up call.

function allocated(f::F, args::Vararg{Any,N}) where {F,N}
    f(args...)
    return @allocated f(args...)
end

@testset "inference" begin
    prob = Rastrigin(6)
    rng = Xoshiro(1)
    alg = MAPElites(prob; dims=(10, 10), batch_size=32)
    state = @inferred init(prob, alg; budget=640, rng)
    emitter = state.emitters[1]
    archive = state.archive
    d = SVector(0.5, -0.5)

    @inferred QD.evaluate(prob, QD.random_solution(rng, prob))
    @inferred QD.random_solution(rng, prob)
    @inferred QD.cell_index(archive, d)
    @inferred QD.sample_elites(rng, archive, 8)
    @inferred QD.ask(rng, emitter, archive)
    @inferred QD.vary(rng, emitter.op, QD.random_solution(rng, prob))
    @inferred QD.batch_size(emitter)
    @inferred length(archive)
    @inferred QD.capacity(archive)
    @inferred coverage(archive)
    @inferred max_fitness(archive)
    @inferred step!(state)
    @inferred solve!(state)
    @inferred QD.evaluate_batch(prob, [QD.random_solution(rng, prob) for _ in 1:4])

    xs = [QD.random_solution(rng, prob) for _ in 1:4]
    evs = QD.evaluate_batch(prob, xs)
    @inferred QD.add!(archive, xs[1], evs[1])
    @inferred QD.add_batch!(archive, xs, evs)
    @inferred QD.tell!(emitter, xs, evs, QD.add_batch!(archive, xs, evs))
end

@testset "inference with a crossover" begin
    prob = Rastrigin(6)
    alg = MAPElites(prob; dims=(10, 10), batch_size=16, variation=IsoLineVariation(0.05, 0.3))
    state = @inferred init(prob, alg; budget=320, rng=Xoshiro(1))
    emitter = state.emitters[1]
    @inferred QD.ask(Xoshiro(1), emitter, state.archive)
    @inferred QD.vary(Xoshiro(1), emitter.op, [0.0, 1.0], [1.0, 0.0])
    @inferred QD.n_parents(emitter.op)
    @inferred step!(state)
end

@testset "inference with several emitters" begin
    prob = Rastrigin(4)
    archive = GridArchive(prob; dims=(5, 5))
    emitters = (VariationEmitter(GaussianMutation(0.1); batch_size=4),
                VariationEmitter(GaussianMutation(1.0); batch_size=3))
    state = init(prob, QDAlgorithm(archive, emitters); budget=100, rng=Xoshiro(1))
    @inferred step!(state)
end

@testset "allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    for outside in (ThrowOutside(), ClampOutside(), RejectOutside())
        archive = GridArchive{SVector{3,Float64}}((10, 10), ((-1, 1), (-1, 1)); outside)
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, ev) == 0                   # same fitness: rejected
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness + 1, ev.descriptor)) == 0  # improvement
        fresh = Evaluation(ev.fitness, SVector(-0.9, 0.9))
        @test allocated(QD.add!, archive, x, fresh) == 0                # new cell
    end

    archive = GridArchive{Vector{Float64}}((10, 10), ((-1, 1), (-1, 1)))
    @test allocated(QD.cell_index, archive, SVector(0.1, 0.2)) == 0
end
