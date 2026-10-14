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

@testset "inference with bounded variation" begin
    prob = Rastrigin(6)
    for variation in (PolynomialMutation(-5.12, 5.12), UniformCrossover())
        alg = MAPElites(prob; dims=(10, 10), batch_size=16, variation)
        state = @inferred init(prob, alg; budget=320, rng=Xoshiro(1))
        @inferred QD.ask(Xoshiro(1), state.emitters[1], state.archive)
        @inferred step!(state)
    end
    @inferred QD.vary(Xoshiro(1), PolynomialMutation(-5.12, 5.12), [0.0, 1.0])
    @inferred QD.vary(Xoshiro(1), PolynomialMutation([-1.0, -2.0], [1.0, 2.0]), [0.0, 1.0])
    @inferred QD.vary(Xoshiro(1), UniformCrossover(), [1, 2], [3, 4])
    # the only allocation is the child
    x = [0.5, -1.0, 2.0]
    child = allocated(n -> Vector{Float64}(undef, n), 3)
    @test allocated(QD.vary, Xoshiro(1), PolynomialMutation(-5.12, 5.12), x) == child
    @test allocated(QD.vary, Xoshiro(1), PolynomialMutation([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]), x) == child
    @test allocated(QD.vary, Xoshiro(1), UniformCrossover(), x, x) == child
end

@testset "inference with CMA emitters" begin
    prob = Rastrigin(6)
    for alg in (CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2),
                CMAMAE(prob; dims=(8, 8), sigma0=1.0, threshold_min=-500.0, learning_rate=0.1, num_emitters=2),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=ObjectiveRanker(), selection=MuSelection()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=TwoStageRandomDirectionRanker()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=RandomDirectionRanker()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=AcceptedRandomDirectionRanker()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, strategy=PyribsCMAES),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=TwoStageObjectiveRanker()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, strategy=SepCMAES),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, strategy=OpenAIES),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, strategy=LMMAES, batch_size=5))
        state = @inferred init(prob, alg; budget=600, rng=Xoshiro(1))
        emitter = state.emitters[1]
        xs = @inferred QD.ask(Xoshiro(1), emitter, state.archive)
        evs = QD.evaluate_batch(prob, xs)
        results = QD.add_batch!(state.archive, xs, evs)
        @inferred QD.tell!(emitter, xs, evs, results)
        @inferred QD.batch_size(emitter)
        @inferred step!(state)
        @inferred solve!(state)
    end
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

@testset "CVTArchive inference and allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    centroids = QD.centroids(CVTArchive(Xoshiro(1), prob; cells=200, samples=5_000))
    for outside in (ThrowOutside(), ClampOutside(), RejectOutside())
        archive = CVTArchive{SVector{3,Float64}}(centroids, ((-1, 1), (-1, 1)); outside)
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @inferred QD.cell_index(archive, ev.descriptor)
        @inferred QD.add!(archive, x, ev)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, ev) == 0                                       # rejected
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness + 1, ev.descriptor)) == 0  # improvement
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness, SVector(-0.9, 0.9))) == 0 # another cell
    end
    archive = CVTArchive(centroids, ((-1, 1), (-1, 1)))
    @inferred QD.sample_elites(Xoshiro(1), (QD.add!(archive, [0.0], Evaluation(1.0, SVector(0.0, 0.0))); archive), 4)
    @inferred max_fitness(archive)
end

@testset "UnstructuredArchive inference and allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    # full and not full, so that both the free-slot and the eviction path run
    for capacity in (200, 5)
        archive = UnstructuredArchive{SVector{3,Float64}}(2; capacity, threshold=0.1)
        for i in 1:20
            QD.add!(archive, QD.random_solution(Xoshiro(i), prob), Evaluation(-i, rand(Xoshiro(i), SVector{2,Float64})))
        end
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @inferred QD.cell_index(archive, ev.descriptor)
        @inferred QD.add!(archive, x, ev)
        @inferred QD.sample_elites(Xoshiro(1), archive, 4)
        @inferred max_fitness(archive)
        @inferred qd_score(archive)
        far = SVector(50.0, 50.0)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, Evaluation(-100.0, far)) == 0           # rejected, or a new slot
        @test allocated(QD.add!, archive, x, Evaluation(1000.0, far)) == 0           # improvement
        @test allocated(QD.add!, archive, x, Evaluation(2000.0, SVector(60.0, 60.0))) == 0   # new slot, or eviction
    end
end

@testset "CategoricalArchive inference and allocations" begin
    labelled = QDProblem(x -> (-sum(abs2, x), (round(Int, clamp(x[1], -3, 3)), x[2] > 0 ? :pos : :neg));
                         init=rng -> SVector{3,Float64}(6 .* rand(rng, 3) .- 3), categories=(-3:3, [:neg, :pos]))
    numeric = QDProblem(x -> (-sum(abs2, x), (round(x[1]), round(x[2])));
                        init=rng -> SVector{3,Float64}(6 .* rand(rng, 3) .- 3), bounds=((-3, 3), (-3, 3)))
    for (prob, make, other) in ((labelled, outside -> CategoricalArchive{SVector{3,Float64}}(QD.descriptor_categories(labelled); outside), (3, :neg)),
                                (numeric, outside -> CategoricalArchive{SVector{3,Float64}}((-3:3, -3:3); outside), SVector(3.0, -3.0))),
        outside in (ThrowOutside(), RejectOutside())
        archive = make(outside)
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @inferred QD.cell_index(archive, ev.descriptor)
        @inferred QD.add!(archive, x, ev)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, ev) == 0                                         # rejected
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness + 1, ev.descriptor)) == 0   # improvement
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness, other)) == 0               # another cell
    end
    archive = CategoricalArchive((-3:3, -3:3))
    @inferred QD.sample_elites(Xoshiro(1), (QD.add!(archive, [0.0], Evaluation(1.0, (0, 0))); archive), 4)
end

@testset "ThresholdArchive inference and allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    for (rate, floor) in ((1.0, -Inf), (1.0, -10.0), (0.3, -10.0))
        archive = ThresholdArchive(GridArchive{SVector{3,Float64}}((10, 10), ((-1, 1), (-1, 1)));
                                   learning_rate=rate, threshold_min=floor)
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @inferred QD.cell_index(archive, ev.descriptor)
        @inferred QD.add!(archive, x, ev)
        @inferred QD.sample_elites(Xoshiro(1), archive, 4)
        @inferred max_fitness(archive)
        @inferred qd_score(archive)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, ev) == 0                                          # rejected
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness + 1, ev.descriptor)) == 0    # improvement
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness, SVector(-0.9, 0.9))) == 0   # new cell
        @test allocated(max_fitness, archive) == 0
        xs = [QD.random_solution(Xoshiro(k), prob) for k in 1:8]
        @inferred QD.add_batch!(archive, xs, QD.evaluate_batch(prob, xs))
    end
end
