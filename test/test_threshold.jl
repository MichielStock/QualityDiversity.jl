include("reference/threshold.jl")

@testset "ThresholdArchive construction" begin
    grid = GridArchive((4, 4), ((0, 1), (0, 1)))
    a = ThresholdArchive(grid)
    @test a isa ThresholdArchive{Vector{Float64},Float64,SVector{2,Float64}}
    @test a isa QD.AbstractArchive{Vector{Float64},Float64,SVector{2,Float64}}
    @test a.inner === grid && a.learning_rate == 1.0 && a.threshold_min == -Inf
    @test QD.capacity(a) == 16 && length(a) == 0 && coverage(a) == 0
    @test QD.thresholds(a) == fill(-Inf, 16)

    b = ThresholdArchive(CVTArchive{Vector{Float32},Float32}(10, ((0, 1), (0, 1)); samples=200, rng=Xoshiro(1));
                         learning_rate=0.25, threshold_min=-3)
    @test b isa ThresholdArchive{Vector{Float32},Float32}
    @test b.learning_rate === 0.25f0 && b.threshold_min === -3.0f0 && QD.thresholds(b) == fill(-3.0f0, 10)
    @test ThresholdArchive(CategoricalArchive((1:3, [:a, :b]))).inner isa CategoricalArchive
    @test ThresholdArchive(GridArchive((2,), ((0, 1),)); threshold_min=0).learning_rate == 1.0

    # the defaults keep the best solutions in the archive itself; otherwise a copy collects them
    @test a.result === nothing && QD.result_archive(a) === a
    c = ThresholdArchive(GridArchive((4, 4), ((0, 1), (0, 1))); learning_rate=0.5, threshold_min=0)
    @test c.result isa GridArchive && c.result !== c.inner && length(c.result) == 0
    @test QD.result_archive(c) === c.result && QD.capacity(c.result) == 16
    @test ThresholdArchive(GridArchive((4, 4), ((0, 1), (0, 1))); threshold_min=0).result isa GridArchive
    own = GridArchive((4, 4), ((0, 1), (0, 1)))
    @test ThresholdArchive(GridArchive((4, 4), ((0, 1), (0, 1))); learning_rate=0.5, threshold_min=0, result=own).result === own
    @test ThresholdArchive(GridArchive((4, 4), ((0, 1), (0, 1))); learning_rate=0.5, threshold_min=0, result=nothing).result === nothing
    cat = ThresholdArchive(CategoricalArchive((1:3, [:a, :b])); learning_rate=0.5, threshold_min=0)
    @test cat.result isa CategoricalArchive && cat.result !== cat.inner

    full = GridArchive((2, 2), ((0, 1), (0, 1)))
    QD.add!(full, [1.0], Evaluation(1.0, SVector(0.1, 0.1)))
    @test_throws "the archive to wrap must be empty but holds 1 solutions" ThresholdArchive(full)
    @test_throws "learning_rate must be in [0, 1], got 1.5" ThresholdArchive(grid; learning_rate=1.5)
    @test_throws "learning_rate must be in [0, 1], got -0.1" ThresholdArchive(grid; learning_rate=-0.1)
    @test_throws "threshold_min must be a number below Inf, got NaN" ThresholdArchive(grid; threshold_min=NaN)
    @test_throws "threshold_min must be a number below Inf, got Inf" ThresholdArchive(grid; threshold_min=Inf)
    @test_throws "threshold_min can only be -Inf if learning_rate is 1, got 0.5; give a finite threshold_min" ThresholdArchive(grid; learning_rate=0.5)
    @test_throws "threshold_min can only be -Inf if learning_rate is 1, got 0; give a finite" ThresholdArchive(grid; learning_rate=0)
    @test_throws MethodError ThresholdArchive(GridArchive{Vector{Float64},Int}((2, 2), ((0, 1), (0, 1))))
    @test_throws "result must be empty but holds 1 solutions" ThresholdArchive(grid; learning_rate=0.5, threshold_min=0, result=full)
    @test_throws "result must be a GridArchive of the same type as the archive it belongs to, got a CVTArchive" ThresholdArchive(
        GridArchive((2, 2), ((0, 1), (0, 1))); learning_rate=0.5, threshold_min=0,
        result=CVTArchive(4, ((0, 1), (0, 1)); samples=100, rng=Xoshiro(1)))
    g = GridArchive((2, 2), ((0, 1), (0, 1)))
    @test_throws "result must be a different archive from the one it belongs to" ThresholdArchive(g; learning_rate=0.5, threshold_min=0, result=g)
end

@testset "ThresholdArchive thresholds and improvement" begin
    ev(f, d=(0.1, 0.1)) = Evaluation(f, SVector{2,Float64}(d))
    a = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1))); learning_rate=0.5, threshold_min=0)
    @test QD.thresholds(a) == zeros(4)

    # an empty cell takes a solution above the minimum; its threshold moves halfway
    r = QD.add!(a, [1.0], ev(-2.0))
    @test r.status == QD.Rejected && r.improvement == -2.0 && length(a) == 0
    @test QD.add!(a, [1.0], ev(0.0)).status == QD.Rejected
    r = QD.add!(a, [1.0], ev(10.0))
    @test r.status == QD.NewCell && r.improvement == 10.0
    @test QD.thresholds(a)[1] == 5.0                      # (1 - 0.5) * 0 + 0.5 * 10

    r = QD.add!(a, [2.0], ev(4.0))                        # below the threshold
    @test r.status == QD.Rejected && r.improvement == -1.0
    r = QD.add!(a, [2.0], ev(5.0))                        # equal to it
    @test r.status == QD.Rejected && r.improvement == 0.0
    @test only(first.(collect(QD.elites(a)))) == [1.0]    # the elite did not change

    r = QD.add!(a, [3.0], ev(8.0))                        # above the threshold, below the elite
    @test r.status == QD.Improved && r.improvement == 3.0
    @test QD.thresholds(a)[1] == 6.5
    @test only(first.(collect(QD.elites(a)))) == [3.0]    # it replaced the better elite
    @test max_fitness(a) == 8.0

    r = QD.add!(a, [4.0], ev(12.0))
    @test r.status == QD.Improved && r.improvement == 5.5
    @test QD.thresholds(a)[1] == 9.25 && max_fitness(a) == 12.0 && qd_score(a) == 12.0

    # above the threshold but below the elite: the elite, and the maximum, get worse
    r = QD.add!(a, [5.0], ev(9.5))
    @test r.status == QD.Improved && r.improvement == 0.25
    @test QD.thresholds(a)[1] == 9.375
    @test only(first.(collect(QD.elites(a)))) == [5.0]
    @test max_fitness(a) == 9.5 && qd_score(a) == 9.5 && mean_fitness(a) == 9.5

    # the result archive holds the best solution of the cell, 12, which the elite no longer is
    best = QD.result_archive(a)
    @test best isa GridArchive && max_fitness(best) == 12.0 && only(first.(collect(QD.elites(best)))) == [4.0]
    @test length(best) == 1

    # another cell is independent
    r = QD.add!(a, [6.0], ev(2.0, (0.9, 0.9)))
    @test r.status == QD.NewCell && r.improvement == 2.0 && QD.thresholds(a)[4] == 1.0
    @test length(a) == 2 && max_fitness(a) == 9.5 && qd_score(a; offset=10) == 9.5 + 2 + 20

    # a solution the archive rejects still reaches the result archive: -2 is below the minimum
    @test length(best) == 2 && QD.add!(a, [7.0], ev(-2.0, (0.1, 0.9))).status == QD.Rejected
    @test length(best) == 3 && length(a) == 2

    empty!(a)
    @test length(a) == 0 && QD.thresholds(a) == zeros(4) && qd_score(a) == 0 && length(best) == 0
    @test_throws "max_fitness is undefined" max_fitness(a)
    @test QD.add!(a, [1.0], ev(1.0)).status == QD.NewCell

    # learning rate 0 keeps the minimum, so every solution above it replaces the elite
    z = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1))); learning_rate=0, threshold_min=1)
    r = QD.add!(z, [1.0], ev(5.0))
    @test r.status == QD.NewCell && r.improvement == 4.0 && QD.thresholds(z)[1] == 1.0
    @test QD.add!(z, [2.0], ev(2.0)).status == QD.Improved
    @test only(first.(collect(QD.elites(z)))) == [2.0]

    # the defaults are the wrapped archive's rules: an empty cell takes anything
    d = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1))))
    r = QD.add!(d, [1.0], ev(-7.0))
    @test r.status == QD.NewCell && r.improvement == -7.0 && QD.thresholds(d)[1] == -7.0
    r = QD.add!(d, [2.0], ev(-3.0))
    @test r.status == QD.Improved && r.improvement == 4.0 && QD.thresholds(d)[1] == -3.0
end

@testset "ThresholdArchive add_batch!" begin
    ev(f, d=(0.1, 0.1)) = Evaluation(f, SVector{2,Float64}(d))
    new_archive(; kwargs...) = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1))); kwargs...)

    # results are computed against the archive before the batch, so all three
    # solutions in the empty cell are new
    a = new_archive(learning_rate=0.5, threshold_min=0)
    rs = QD.add_batch!(a, [[1.0], [2.0], [3.0]], [ev(10.0), ev(6.0), ev(3.0)])
    @test [r.status for r in rs] == fill(QD.NewCell, 3)
    @test [r.improvement for r in rs] == [10.0, 6.0, 3.0]
    @test only(first.(collect(QD.elites(a)))) == [1.0]                      # the highest fitness is the elite
    @test QD.thresholds(a)[1] ≈ 0.875 * (19 / 3)                           # (1 - 0.5^3) * mean(10, 6, 3)
    @test length(a) == 1 && max_fitness(a) == 10.0

    # a threshold of 5, and a batch of three: one is rejected, two pass
    a = new_archive(learning_rate=0.5, threshold_min=0)
    QD.add!(a, [0.0], ev(10.0))
    rs = QD.add_batch!(a, [[1.0], [2.0], [3.0]], [ev(4.0), ev(8.0), ev(12.0)])
    @test [r.status for r in rs] == [QD.Rejected, QD.Improved, QD.Improved]
    @test [r.improvement for r in rs] == [-1.0, 3.0, 7.0]
    @test only(first.(collect(QD.elites(a)))) == [3.0]
    @test QD.thresholds(a)[1] == 0.25 * 5 + 0.75 * 10                        # 0.5^2 * 5 + (1 - 0.5^2) * mean(8, 12)
    @test max_fitness(a) == 12.0

    # a tie goes to the first solution; separate cells are independent
    a = new_archive(learning_rate=0.5, threshold_min=0)
    rs = QD.add_batch!(a, [[1.0], [2.0], [3.0], [4.0]],
                       [ev(9.0), ev(9.0), ev(1.0, (0.9, 0.9)), ev(-1.0, (0.9, 0.9))])
    @test [r.status for r in rs] == [QD.NewCell, QD.NewCell, QD.NewCell, QD.Rejected]
    @test sort(first.(collect(QD.elites(a)))) == [[1.0], [3.0]]
    @test QD.thresholds(a)[[1, 4]] == [0.75 * 9, 0.5]

    # the thresholds-as-fitness setting uses the highest fitness, as MAP-Elites does
    a = new_archive()
    rs = QD.add_batch!(a, [[1.0], [2.0], [3.0]], [ev(3.0), ev(7.0), ev(5.0)])
    @test [r.status for r in rs] == fill(QD.NewCell, 3) && [r.improvement for r in rs] == [3.0, 7.0, 5.0]
    @test QD.thresholds(a)[1] == 7.0 && only(first.(collect(QD.elites(a)))) == [2.0]
    rs = QD.add_batch!(a, [[4.0], [5.0]], [ev(6.0), ev(8.0)])
    @test [r.status for r in rs] == [QD.Rejected, QD.Improved] && [r.improvement for r in rs] == [-1.0, 1.0]
    @test QD.thresholds(a)[1] == 8.0

    # a batch of one is a single add
    rng = Xoshiro(4)
    one_by_one = new_archive(learning_rate=0.3, threshold_min=-50)
    batched = new_archive(learning_rate=0.3, threshold_min=-50)
    for k in 1:500
        e = ev(-60 * rand(rng), tuple(rand(rng, 2)...))
        @test QD.add!(one_by_one, [Float64(k)], e) == only(QD.add_batch!(batched, [[Float64(k)]], [e]))
    end
    @test QD.thresholds(batched) == QD.thresholds(one_by_one)
    @test collect(QD.elites(batched)) == collect(QD.elites(one_by_one))

    # the result of a solution, and the archive after the batch, do not depend on the batch order
    xs = [[Float64(k)] for k in 1:60]
    evs = [ev(-40 * rand(rng), tuple(rand(rng, 2)...)) for _ in 1:60]
    perm = randperm(rng, 30)
    a = new_archive(learning_rate=0.2, threshold_min=-50)
    b = new_archive(learning_rate=0.2, threshold_min=-50)
    QD.add_batch!(a, xs[1:30], evs[1:30]); QD.add_batch!(b, xs[1:30], evs[1:30])
    ra = QD.add_batch!(a, xs[31:60], evs[31:60])
    rb = QD.add_batch!(b, xs[31:60][perm], evs[31:60][perm])
    @test ra[perm] == rb
    @test QD.thresholds(a) ≈ QD.thresholds(b)
    @test sort(first.(collect(QD.elites(a)))) == sort(first.(collect(QD.elites(b))))

    # the result archive receives every solution of the batch
    a = new_archive(learning_rate=0.5, threshold_min=0)
    QD.add_batch!(a, [[1.0], [2.0], [3.0], [4.0]], [ev(10.0), ev(6.0), ev(-1.0), ev(3.0, (0.9, 0.9))])
    @test sort(first.(collect(QD.elites(QD.result_archive(a))))) == [[1.0], [4.0]]
    @test length(QD.result_archive(a)) == 2 && length(a) == 2
    @test max_fitness(QD.result_archive(a)) == 10.0

    # rejected descriptors, errors
    r = new_archive()
    @test_throws "got 2 solutions but 1 evaluations" QD.add_batch!(r, [[1.0], [2.0]], [ev(1.0)])
    discarding = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1)); outside=RejectOutside()))
    rs = QD.add_batch!(discarding, [[1.0], [2.0]], [ev(1.0, (5.0, 0.5)), ev(2.0)])
    @test [r.status for r in rs] == [QD.Rejected, QD.NewCell] && rs[1].improvement == -Inf
    @test isempty(QD.add_batch!(r, Vector{Float64}[], Evaluation{Float64,SVector{2,Float64}}[]))
end

@testset "ThresholdArchive with learning rate 1 is the wrapped archive" begin
    rng = Xoshiro(3)
    for make in (() -> GridArchive((6, 6), ((0, 1), (0, 1))),
                 () -> CVTArchive(30, ((0, 1), (0, 1)); samples=1_000, rng=Xoshiro(1)))
        plain = make()
        wrapped = ThresholdArchive(make())
        for k in 1:3_000
            x = [Float64(k)]
            e = Evaluation(-10 * rand(rng), rand(rng, SVector{2,Float64}))
            @test QD.add!(plain, x, e) == QD.add!(wrapped, x, e)
        end
        @test length(wrapped) == length(plain)
        @test collect(QD.elites(wrapped)) == collect(QD.elites(plain))
        @test max_fitness(wrapped) == max_fitness(plain)
        @test qd_score(wrapped; offset=10) == qd_score(plain; offset=10)
        @test QD.thresholds(wrapped)[plain.occupied_idx] == plain.fitness[plain.occupied_idx]
        # a batch ends in the same archive, though its results are against the archive before it
        xs = [[Float64(k)] for k in 1:400]
        evs = [Evaluation(-10 * rand(rng), rand(rng, SVector{2,Float64})) for _ in 1:400]
        QD.add_batch!(wrapped, xs, evs)
        QD.add_batch!(plain, xs, evs)
        @test collect(QD.elites(wrapped)) == collect(QD.elites(plain))
    end
end

@testset "ThresholdArchive running metrics with decreasing elites" begin
    rng = Xoshiro(11)
    for rate in (0.1, 0.5, 1.0)
        a = ThresholdArchive(GridArchive((8, 8), ((0, 1), (0, 1))); learning_rate=rate, threshold_min=-200)
        plain = GridArchive((8, 8), ((0, 1), (0, 1)))
        for round in 1:20, k in 1:30
            evs = [Evaluation(-100 * rand(rng), rand(rng, SVector{2,Float64})) for _ in 1:10]
            xs = [[Float64(round), Float64(k), Float64(j)] for j in 1:10]
            isodd(k) ? QD.add_batch!(a, xs, evs) : foreach((x, e) -> QD.add!(a, x, e), xs, evs)
            QD.add_batch!(plain, xs, evs)
            if round % 5 == 0 && k == 30       # the metrics can be read at any time
                @test max_fitness(a) == generic_max_fitness(a)
                @test qd_score(a; offset=250.5) ≈ generic_qd_score(a; offset=250.5) rtol = 1e-14
                @test mean_fitness(a) ≈ generic_mean_fitness(a) rtol = 1e-14
            end
        end
        @test coverage(a) == length(a) / QD.capacity(a)
        # the result archive is the archive that MAP-Elites would hold
        @test collect(QD.elites(QD.result_archive(a))) == collect(QD.elites(plain))
        @test max_fitness(QD.result_archive(a)) >= max_fitness(a)
        # every threshold is at most the best fitness that ever entered, and elites exceed no threshold by much
        @test all(<=(0), QD.thresholds(a))
    end
    # the maximum drops when its elite is replaced by a worse solution, and recovers
    a = ThresholdArchive(GridArchive((2, 1), ((0, 1), (0, 1))); learning_rate=0.5, threshold_min=0)
    ev(f, d) = Evaluation(f, SVector{2,Float64}(d))
    QD.add!(a, [1.0], ev(10.0, (0.2, 0.5)))
    QD.add!(a, [2.0], ev(20.0, (0.2, 0.5)))             # threshold 12.5, elite 20
    QD.add!(a, [3.0], ev(5.0, (0.8, 0.5)))
    @test max_fitness(a) == 20.0
    QD.add!(a, [4.0], ev(16.0, (0.2, 0.5)))             # threshold 12.5: replaces the best elite
    @test max_fitness(a) == 16.0 && qd_score(a) == 21.0
    QD.add!(a, [5.0], ev(30.0, (0.8, 0.5)))
    @test max_fitness(a) == 30.0
    QD.add!(a, [6.0], ev(1.0, (0.2, 0.5)))              # below the threshold 14.25: rejected
    @test max_fitness(a) == 30.0
end

@testset "ThresholdArchive rejects what the wrapped archive discards" begin
    a = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1)); outside=RejectOutside()))
    r = QD.add!(a, [1.0], Evaluation(1.0, SVector(5.0, 0.5)))
    @test r.status == QD.Rejected && r.improvement == -Inf && length(a) == 0
    @test QD.cell_index(a, SVector(5.0, 0.5)) == 0
    @test_throws "descriptor has 3 dimensions but the archive has 2" QD.cell_index(a, [0.0, 0.0, 0.0])
    t = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1))))
    @test_throws "outside the archive bounds" QD.add!(t, [1.0], Evaluation(1.0, SVector(5.0, 0.5)))
    @test_throws "cannot sample elites from an empty archive" QD.sample_elites(Xoshiro(1), t, 3)
end

@testset "MAPElites over a ThresholdArchive" begin
    prob = Rastrigin(6)
    archive = ThresholdArchive(GridArchive(prob; dims=(10, 10)); learning_rate=0.3, threshold_min=-400)
    alg = MAPElites(archive; batch_size=50, variation=IsoLineVariation(0.05, 0.3))
    result = solve(prob, alg; budget=6_000, rng=Xoshiro(1))
    @test result.n_evals == 6_000
    # a run returns the archive of the best solutions, and reports its metrics
    @test result.archive isa GridArchive && length(archive) == 0 && length(archive.result) == 0
    @test length(result.archive) > 30
    @test issorted(r.max_fitness for r in result.history)        # the soft archive's maximum can fall
    @test issorted(r.qd_score for r in result.history)
    @test result.history[end].max_fitness == max_fitness(result.archive)
    @test result.history[end].qd_score == qd_score(result.archive; offset=QD.fitness_offset(prob))
    @test solve(prob, alg; budget=6_000, rng=Xoshiro(1)).history == result.history

    # stepping by hand: the working archive is the threshold archive
    state = init(prob, alg; budget=1_000, rng=Xoshiro(1))
    @test state.archive isa ThresholdArchive && QD.result_archive(state.archive) isa GridArchive
    @test length(QD.result_archive(state.archive)) >= length(state.archive)

    # with the defaults the archive is its own result and the run matches plain MAP-Elites
    own = MAPElites(ThresholdArchive(GridArchive(prob; dims=(10, 10))); batch_size=50, variation=IsoLineVariation(0.05, 0.3))
    plain = MAPElites(prob; dims=(10, 10), batch_size=50, variation=IsoLineVariation(0.05, 0.3))
    a = solve(prob, own; budget=3_000, rng=Xoshiro(2))
    b = solve(prob, plain; budget=3_000, rng=Xoshiro(2))
    @test a.archive isa ThresholdArchive
    # the metrics agree up to the order in which the running sum was accumulated
    @test length(a.history) == length(b.history)
    @test all(i -> all(k -> isapprox(a.history[i][k], b.history[i][k]; rtol=1e-12), keys(a.history[i])),
              eachindex(a.history))

    wrong = ThresholdArchive(GridArchive((3,), ((0, 1),)))
    @test_throws "the problem has 2 descriptor dimensions but the archive has 1" init(prob, MAPElites(wrong); budget=100)
end

@testset "ThresholdArchive matches pyribs" begin
    # reference/threshold.jl: the statuses, improvements, thresholds and elites of
    # pyribs's GridArchive with a learning rate after each batch of a sequence
    pyribs_status = Dict(0 => QD.Rejected, 1 => QD.Improved, 2 => QD.NewCell)
    for case in threshold_cases
        a = ThresholdArchive(GridArchive((case.cells,), ((0, 1),));
                             learning_rate=case.learning_rate, threshold_min=case.threshold_min)
        total = length(case.fitness)
        statuses = QD.AddStatus[]
        values = Float64[]
        for b in 1:total÷case.batch
            ids = (b-1)*case.batch+1:b*case.batch
            rs = QD.add_batch!(a, [[Float64(i)] for i in ids],
                               [Evaluation(case.fitness[i], SVector(case.descriptors[i])) for i in ids])
            append!(statuses, [r.status for r in rs])
            append!(values, [r.improvement for r in rs])
            after = (b-1)*case.cells+1:b*case.cells
            @test QD.thresholds(a) ≈ case.thresholds[after] rtol = 1e-12
            kept = zeros(Int, case.cells)
            for (x, ev) in QD.elites(a)
                kept[QD.cell_index(a, ev.descriptor)] = Int(only(x))
            end
            @test kept == case.elites[after]
        end
        @test statuses == [pyribs_status[s] for s in case.statuses]
        @test values ≈ case.values rtol = 1e-12
    end
end
