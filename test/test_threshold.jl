@testset "ThresholdArchive construction" begin
    grid = GridArchive((4, 4), ((0, 1), (0, 1)))
    a = ThresholdArchive(grid)
    @test a isa ThresholdArchive{Vector{Float64},Float64,SVector{2,Float64}}
    @test a isa QD.AbstractArchive{Vector{Float64},Float64,SVector{2,Float64}}
    @test a.inner === grid && a.learning_rate == 1.0 && a.threshold_min == -Inf
    @test QD.capacity(a) == 16 && length(a) == 0 && coverage(a) == 0
    @test QD.thresholds(a) == fill(-Inf, 16)

    b = ThresholdArchive(CVTArchive{Vector{Float32},Float32}(Xoshiro(1), 10, ((0, 1), (0, 1)); samples=200);
                         learning_rate=0.25, threshold_min=-3)
    @test b isa ThresholdArchive{Vector{Float32},Float32}
    @test b.learning_rate === 0.25f0 && b.threshold_min === -3.0f0 && QD.thresholds(b) == fill(-3.0f0, 10)
    @test ThresholdArchive(CategoricalArchive((1:3, [:a, :b]))).inner isa CategoricalArchive
    @test ThresholdArchive(GridArchive((2,), ((0, 1),)); threshold_min=0).learning_rate == 1.0

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

    # another cell is independent
    r = QD.add!(a, [6.0], ev(2.0, (0.9, 0.9)))
    @test r.status == QD.NewCell && r.improvement == 2.0 && QD.thresholds(a)[4] == 1.0
    @test length(a) == 2 && max_fitness(a) == 9.5 && qd_score(a; offset=10) == 9.5 + 2 + 20

    empty!(a)
    @test length(a) == 0 && QD.thresholds(a) == zeros(4) && qd_score(a) == 0
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

    # rejected descriptors, errors
    r = new_archive()
    @test_throws "got 2 solutions but 1 evaluations" QD.add_batch!(r, [[1.0], [2.0]], [ev(1.0)])
    discarding = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1)); outside=RejectOutside()))
    rs = QD.add_batch!(discarding, [[1.0], [2.0]], [ev(1.0, (5.0, 0.5)), ev(2.0)])
    @test [r.status for r in rs] == [QD.Rejected, QD.NewCell] && rs[1].improvement == 0.0
    @test isempty(QD.add_batch!(r, Vector{Float64}[], Evaluation{Float64,SVector{2,Float64}}[]))
end

@testset "ThresholdArchive with learning rate 1 is the wrapped archive" begin
    rng = Xoshiro(3)
    for make in (() -> GridArchive((6, 6), ((0, 1), (0, 1))),
                 () -> CVTArchive(Xoshiro(1), 30, ((0, 1), (0, 1)); samples=1_000))
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
        for round in 1:20, k in 1:30
            evs = [Evaluation(-100 * rand(rng), rand(rng, SVector{2,Float64})) for _ in 1:10]
            isodd(k) ? QD.add_batch!(a, fill([0.0], 10), evs) : foreach(e -> QD.add!(a, [0.0], e), evs)
            if round % 5 == 0 && k == 30       # the metrics can be read at any time
                @test max_fitness(a) == generic_max_fitness(a)
                @test qd_score(a; offset=250.5) ≈ generic_qd_score(a; offset=250.5) rtol = 1e-14
                @test mean_fitness(a) ≈ generic_mean_fitness(a) rtol = 1e-14
            end
        end
        @test coverage(a) == length(a) / QD.capacity(a)
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
    @test r.status == QD.Rejected && r.improvement == 0.0 && length(a) == 0
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
    @test result.archive isa ThresholdArchive && length(archive) == 0
    @test length(result.archive) > 30
    @test solve(prob, alg; budget=6_000, rng=Xoshiro(1)).history == result.history
    @test all(r -> isfinite(r.max_fitness), result.history)
    @test result.archive.inner !== archive.inner       # the run works on a copy

    wrong = ThresholdArchive(GridArchive((3,), ((0, 1),)))
    @test_throws "the problem has 2 descriptor dimensions but the archive has 1" init(prob, MAPElites(wrong); budget=100)
end
