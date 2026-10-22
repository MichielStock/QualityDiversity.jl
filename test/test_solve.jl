@testset "MAPElites on Rastrigin" begin
    prob = Rastrigin(6)
    alg = MAPElites(prob; dims=(10, 10), batch_size=50, variation=GaussianMutation(0.5))
    @test alg isa QDAlgorithm
    @test length(alg.archive) == 0

    result = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test result isa QDResult
    @test result.n_evals == 5_000
    @test length(alg.archive) == 0   # the algorithm is not modified
    @test length(result.archive) > 10
    @test coverage(result.archive) == length(result.archive) / 100
    @test length(result.history) == 1 + (5_000 - 50) ÷ 50 + 0
    @test issorted(r.n_evals for r in result.history)
    @test issorted(r.qd_score for r in result.history)
    @test result.history[end].coverage == coverage(result.archive)
    @test all(ev.fitness <= 0 for (_, ev) in QD.elites(result.archive))
    @test all(ev.fitness + QD.fitness_offset(prob) >= 0 for (_, ev) in QD.elites(result.archive))

    # a run is reproducible from its seed
    again = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test again.history == result.history
    other = solve(prob, alg; budget=5_000, rng=Xoshiro(2))
    @test other.history != result.history
end

@testset "stepping and callbacks" begin
    prob = Rastrigin(4)
    alg = MAPElites(prob; dims=(5, 5), batch_size=10)
    calls = Int[]
    state = init(prob, alg; budget=100, rng=Xoshiro(1), callback=s -> push!(calls, s.n_evals))
    @test state isa QDState
    @test state.n_evals == 10        # initial population: one batch
    @test calls == [10]
    step!(state)
    @test state.n_evals == 20
    @test calls == [10, 20]
    @test length(state.history) == 2
    result = solve!(state)
    @test result.n_evals == 100
    @test solve!(state).n_evals == 100

    state = init(prob, alg; budget=100, rng=Xoshiro(1), init_size=25)
    @test state.n_evals == 25
    # the budget is checked after each step, so a run can overshoot by a batch
    @test solve!(state).n_evals == 105
end

@testset "init validates its arguments" begin
    prob = Rastrigin(4)
    alg = MAPElites(prob; dims=(5, 5), batch_size=10)
    @test_throws "budget (5) is smaller than the initial population (10)" init(prob, alg; budget=5)
    @test_throws "init_size must be at least 1, got 0" init(prob, alg; budget=5, init_size=0)
    # a budget of exactly the initial population is spent by init
    state = init(prob, alg; budget=10, rng=Xoshiro(1))
    @test state.n_evals == 10
    @test solve!(state).n_evals == 10
    # an initial population that the archive discards entirely
    outside = QDProblem(x -> (1.0, (5.0,)); init=rng -> [0.0], bounds=((0, 1),))
    discarding = MAPElites(GridArchive((4,), ((0, 1),); outside=RejectOutside()))
    @test_throws "none of the 64 initial solutions entered the archive" init(outside, discarding; budget=1_000)
end

@testset "several emitters" begin
    prob = Rastrigin(4)
    archive = GridArchive(prob; dims=(5, 5))
    emitters = (VariationEmitter(GaussianMutation(0.1); batch_size=5),
                VariationEmitter(GaussianMutation(1.0); batch_size=3))
    state = init(prob, QDAlgorithm(archive, emitters); budget=50, rng=Xoshiro(1))
    @test state.n_evals == 8
    step!(state)
    @test state.n_evals == 16
end

@testset "QDProblem end to end" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2])); init=rng -> 2 .* rand(rng, 3) .- 1,
                     bounds=((-100, 100), (-100, 100)))
    result = solve(prob, MAPElites(prob; dims=(4, 4), batch_size=20); budget=400, rng=Xoshiro(3))
    @test result.n_evals == 400
    @test length(result.archive) >= 1
end

@testset "the genotype type is that of the random solutions" begin
    f(x) = (-sum(abs2, x), clamp.((x[1], x[2]), -3, 3))
    bounds = ((-3, 3), (-3, 3))
    static = QDProblem(f; init=rng -> SVector{3,Float64}(randn(rng, 3)), bounds)
    @test QD.default_genotype(static) == SVector{3,Float64}
    @test QD.default_genotype(Rastrigin(4)) == Vector{Float64}
    result = solve(static, MAPElites(static; dims=(4, 4), batch_size=8); budget=80, rng=Xoshiro(1))
    @test result.archive isa GridArchive{SVector{3,Float64}}
    single = QDProblem(f; init=rng -> randn(rng, Float32, 3), bounds)
    state = init(single, MAPElites(single; dims=(4, 4), batch_size=8, variation=GaussianMutation(0.1f0));
                 budget=80, rng=Xoshiro(1))
    @test state.archive isa GridArchive{Vector{Float32}}
    @test eltype(QD.ask(Xoshiro(1), state.emitters[1], state.archive)[1]) == Float32
    # a genotype that is not a real vector needs only a variation operator
    struct Shuffle <: AbstractVariation end
    QD.vary(rng, ::Shuffle, parent::String) = String(parent[randperm(rng, length(parent))])
    words = QDProblem(s -> (-Float64(count('a', s)), (clamp(count('b', s), 0, 5), 1.0));
                      init=rng -> String(rand(rng, 'a':'c', 6)), bounds=((0, 5), (0, 2)))
    alg = MAPElites(words; dims=(6, 2), variation=Shuffle(), batch_size=8)
    @test alg.archive isa GridArchive{String}
    @test solve(words, alg; budget=80, rng=Xoshiro(1)).archive isa GridArchive{String}
    # every constructor that takes a problem agrees, and `genotype` still overrides
    for a in (GridArchive(static; dims=(4, 4)), SlidingBoundariesArchive(static; dims=(4, 4)),
              CVTArchive(Xoshiro(1), static; cells=5), UnstructuredArchive(static; capacity=5, threshold=0.1),
              DNSArchive(static; capacity=5, k=2), CMAME(static; dims=(4, 4), sigma0=1.0).archive)
        @test a isa QD.AbstractArchive{SVector{3,Float64}}
    end
    @test GridArchive(static; dims=(4, 4), genotype=Vector{Float64}) isa GridArchive{Vector{Float64}}
end

@testset "type stability" begin
    prob = Rastrigin(4)
    alg = MAPElites(prob; dims=(5, 5), batch_size=10)
    state = @inferred init(prob, alg; budget=100, rng=Xoshiro(1))
    @inferred step!(state)
    @inferred solve!(state)
    @inferred QD.evaluate(prob, zeros(4))
    @inferred QD.cell_index(state.archive, SVector(0.0, 0.0))
    @inferred QD.sample_elites(Xoshiro(1), state.archive, 3)
    @inferred QD.ask(Xoshiro(1), state.emitters[1], state.archive)
end

@testset "bounds and outside policy through MAPElites" begin
    prob = Rastrigin(4)
    alg = MAPElites(prob; dims=(5, 5), bounds=((-1, 1), (-1, 1)), batch_size=20, outside=RejectOutside())
    @test alg.archive.upper == SVector(1.0, 1.0)
    result = solve(prob, alg; budget=2_000, rng=Xoshiro(1))
    @test result.n_evals == 2_000
    @test 0 < length(result.archive) <= 25

    # with the default policy, a narrower archive fails on the first descriptor outside it
    narrow = MAPElites(prob; dims=(5, 5), bounds=((-1, 1), (-1, 1)), batch_size=20)
    @test_throws "outside the archive bounds" solve(prob, narrow; budget=2_000, rng=Xoshiro(1))
end

@testset "init checks the archive against the problem" begin
    prob = Rastrigin(4)
    alg = QDAlgorithm(GridArchive((3,), ((-5, 5),)), (VariationEmitter(GaussianMutation(0.1); batch_size=5),))
    @test_throws "the problem has 2 descriptor dimensions but the archive has 1" init(prob, alg; budget=20)
end

@testset "MAP-Elites with IsoLineVariation" begin
    prob = Rastrigin(6)
    alg = MAPElites(prob; dims=(10, 10), batch_size=50, variation=IsoLineVariation(0.05, 0.3))
    @test alg.emitters[1].op isa IsoLineVariation
    result = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test result.n_evals == 5_000
    @test length(result.archive) > 10
    @test issorted(r.qd_score for r in result.history)
    @test solve(prob, alg; budget=5_000, rng=Xoshiro(1)).history == result.history
    # it differs from a mutation with the same seed
    mutated = solve(prob, MAPElites(prob; dims=(10, 10), batch_size=50); budget=5_000, rng=Xoshiro(1))
    @test mutated.history != result.history
end

@testset "display" begin
    prob = Rastrigin(4)
    alg = MAPElites(prob; dims=(5, 5), batch_size=10)
    result = solve(prob, alg; budget=100, rng=Xoshiro(1))
    n = length(result.archive)
    @test repr(MIME"text/plain"(), result.archive) == "GridArchive with $n of 25 cells occupied"
    shown = repr(MIME"text/plain"(), result)
    @test startswith(shown, "QDResult after 100 evaluations\n  archive: GridArchive with $n of 25 cells occupied\n")
    @test occursin("\n  coverage: $(n / 25)", shown)
    @test !occursin("n_evals", shown)

    state = init(prob, alg; budget=100, rng=Xoshiro(1))
    @test startswith(repr(MIME"text/plain"(), state), "QDState after 10 of 100 evaluations\n  archive: GridArchive with ")
    # compact forms, for `print` and inside collections
    @test repr(result) == "QDResult(100 evaluations, $n cells occupied)"
    @test repr(state) == "QDState(10 of 100 evaluations, $(length(state.archive)) cells occupied)"
end
