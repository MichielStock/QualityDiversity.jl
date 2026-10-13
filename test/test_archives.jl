@testset "GridArchive construction" begin
    a = GridArchive((4, 5), ((0, 1), (-1, 1)))
    @test a isa GridArchive{Vector{Float64},Float64,2}
    @test a isa AbstractArchive{Vector{Float64},Float64,SVector{2,Float64}}
    @test QD.capacity(a) == 20
    @test length(a) == 0
    @test GridArchive{Int}((2,), ((0, 1),)) isa GridArchive{Int,Float64,1}
    @test GridArchive{Int,Float32}((2,), ((0, 1),)) isa GridArchive{Int,Float32,1}
    @test GridArchive(Rastrigin(4); dims=(3, 3)) isa GridArchive{Vector{Float64},Float64,2}

    @test_throws "dims has 2 entries but bounds has 1" GridArchive((2, 2), ((0, 1),))
    @test_throws "dims[2] is 0" GridArchive((2, 0), ((0, 1), (0, 1)))
    @test_throws "bounds[1] is (1.0, 1.0)" GridArchive((2,), ((1, 1),))
    @test_throws "dims must have at least one entry" GridArchive((), ())
end

@testset "GridArchive.cell_index" begin
    a = GridArchive((4, 5), ((0, 1), (-1, 1)))
    @test QD.cell_index(a, [0.0, -1.0]) == 1
    @test QD.cell_index(a, [0.3, -1.0]) == 2
    @test QD.cell_index(a, [0.0, -0.5]) == 5
    @test QD.cell_index(a, [1.0, 1.0]) == 20     # upper bound is in the last cell
    @test QD.cell_index(a, SVector(0.99, 0.99)) == 20
    all_cells = [QD.cell_index(a, [x, y]) for x in 0.125:0.25:1, y in -0.8:0.4:1]
    @test sort(vec(all_cells)) == 1:20

    @test_throws "descriptor has 3 dimensions but the archive has 2" QD.cell_index(a, [0.0, 0.0, 0.0])
    @test_throws "descriptor component 1 is 1.5, outside the archive bounds [0.0, 1.0]" QD.cell_index(a, [1.5, 0.0])
    @test_throws "descriptor component 2 is NaN" QD.cell_index(a, [0.5, NaN])
end

@testset "GridArchive.add!" begin
    a = GridArchive((2, 2), ((0, 1), (0, 1)))
    ev(f, d) = Evaluation(f, SVector{2,Float64}(d))

    r = QD.add!(a, [1.0], ev(-3.0, (0.1, 0.1)))
    @test r.status == QD.NewCell
    @test r.improvement == -3.0
    @test length(a) == 1

    r = QD.add!(a, [2.0], ev(-1.0, (0.2, 0.2)))
    @test r.status == QD.Improved
    @test r.improvement == 2.0
    @test length(a) == 1

    r = QD.add!(a, [3.0], ev(-2.0, (0.2, 0.2)))
    @test r.status == QD.Rejected
    @test r.improvement == -1.0
    r = QD.add!(a, [3.0], ev(-1.0, (0.2, 0.2)))
    @test r.status == QD.Rejected
    @test r.improvement == 0.0

    @test only(first.(collect(QD.elites(a)))) == [2.0]

    QD.add!(a, [4.0], ev(5.0, (0.9, 0.9)))
    @test length(a) == 2
    @test QD.coverage(a) == 0.5
    @test_throws "outside the archive bounds" QD.add!(a, [0.0], ev(0.0, (2.0, 0.0)))
    @test_throws "descriptor has 1 dimensions but the archive has 2" QD.add!(a, [0.0], Evaluation(0.0, SVector(0.5)))

    # fitness is converted to the archive's fitness type
    b = GridArchive{Vector{Float64},Float32}((1,), ((0, 1),))
    @test QD.add!(b, [0.0], Evaluation(1, SVector(0.5))).improvement === 1.0f0

    empty!(a)
    @test length(a) == 0
    @test QD.add!(a, [1.0], ev(-3.0, (0.1, 0.1))).status == QD.NewCell
end

@testset "GridArchive.add_batch!" begin
    a = GridArchive((2,), ((0, 1),))
    evs = [Evaluation(1.0, SVector(0.1)), Evaluation(2.0, SVector(0.2)), Evaluation(1.0, SVector(0.9))]
    rs = QD.add_batch!(a, [[1.0], [2.0], [3.0]], evs)
    @test [r.status for r in rs] == [QD.NewCell, QD.Improved, QD.NewCell]
    @test rs isa Vector{AddResult{Float64}}
    @test_throws "got 1 solutions but 3 evaluations" QD.add_batch!(a, [[1.0]], evs)
end

@testset "GridArchive.sample_elites" begin
    a = GridArchive((10,), ((0, 1),))
    @test_throws "cannot sample elites from an empty archive" QD.sample_elites(Xoshiro(1), a, 3)
    QD.add!(a, [1.0], Evaluation(0.0, SVector(0.05)))
    QD.add!(a, [2.0], Evaluation(0.0, SVector(0.55)))
    parents = QD.sample_elites(Xoshiro(1), a, 200)
    @test parents isa Vector{Vector{Float64}}
    @test Set(parents) == Set([[1.0], [2.0]])
    @test QD.sample_elites(Xoshiro(2), a, 20) == QD.sample_elites(Xoshiro(2), a, 20)
end

@testset "metrics" begin
    a = GridArchive((4,), ((0, 1),))
    @test QD.coverage(a) == 0.0
    @test_throws "max_fitness is undefined for an archive with no elites" max_fitness(a)
    @test_throws "mean_fitness is undefined" mean_fitness(a)
    QD.add!(a, [0.0], Evaluation(-2.0, SVector(0.1)))
    QD.add!(a, [0.0], Evaluation(-4.0, SVector(0.6)))
    @test coverage(a) == 0.5
    @test max_fitness(a) == -2.0
    @test mean_fitness(a) == -3.0
    @test qd_score(a) == -6.0
    @test qd_score(a; offset=10) == 14.0
end

@testset "GridArchive outside policies" begin
    bounds = ((0, 1), (0, 1))
    ev(f, d) = Evaluation(f, SVector{2,Float64}(d))

    @test GridArchive((2, 2), bounds) isa GridArchive{Vector{Float64},Float64,2,ThrowOutside}
    @test_throws "outside the archive bounds" QD.cell_index(GridArchive((2, 2), bounds), [1.5, 0.5])

    clamped = GridArchive((2, 2), bounds; outside=ClampOutside())
    @test QD.cell_index(clamped, [-3.0, 0.2]) == QD.cell_index(clamped, [0.0, 0.2])
    @test QD.cell_index(clamped, [9.0, 9.0]) == 4
    @test QD.add!(clamped, [1.0], ev(1.0, (2.0, 2.0))).status == QD.NewCell
    @test_throws "descriptor component 2 is NaN and cannot be clamped" QD.cell_index(clamped, [0.5, NaN])

    rejecting = GridArchive{Vector{Float64},Float32}((2, 2), bounds; outside=RejectOutside())
    @test QD.cell_index(rejecting, [1.5, 0.5]) == 0
    @test QD.cell_index(rejecting, [0.5, NaN]) == 0
    r = QD.add!(rejecting, [1.0], ev(1.0, (1.5, 0.5)))
    @test r.status == QD.Rejected
    @test r.improvement === 0.0f0
    @test length(rejecting) == 0
    @test QD.add!(rejecting, [1.0], ev(1.0, (0.5, 0.5))).status == QD.NewCell

    # every constructor form takes the policy
    @test GridArchive{Vector{Float64}}((2, 2), bounds; outside=ClampOutside()) isa GridArchive{Vector{Float64},Float64,2,ClampOutside}
    @test GridArchive(Rastrigin(4); dims=(2, 2), outside=RejectOutside()) isa GridArchive{Vector{Float64},Float64,2,RejectOutside}
end

@testset "GridArchive bounds override" begin
    prob = Rastrigin(4)
    a = GridArchive(prob; dims=(2, 2), bounds=((-1, 1), (-2, 2)))
    @test a.lower == SVector(-1.0, -2.0)
    @test a.upper == SVector(1.0, 2.0)
    @test GridArchive(prob; dims=(2, 2)).upper == SVector(5.12, 5.12)
end

# The reference for the running metrics: the generic methods, which make a
# fresh pass over the elites.
generic_qd_score(a; offset) = invoke(qd_score, Tuple{QD.AbstractArchive}, a; offset)
generic_max_fitness(a) = invoke(max_fitness, Tuple{QD.AbstractArchive}, a)
generic_mean_fitness(a) = invoke(mean_fitness, Tuple{QD.AbstractArchive}, a)

@testset "GridArchive running metrics" begin
    rng = Xoshiro(11)
    a = GridArchive((20, 20), ((0, 1), (0, 1)))
    function check(a)
        @test coverage(a) == length(a) / QD.capacity(a)
        @test qd_score(a; offset=0) ≈ generic_qd_score(a; offset=0) rtol = 1e-14
        @test qd_score(a; offset=250.5) ≈ generic_qd_score(a; offset=250.5) rtol = 1e-14
        @test max_fitness(a) == generic_max_fitness(a)
        @test mean_fitness(a) ≈ generic_mean_fitness(a) rtol = 1e-14
    end

    # new cells, improvements and rejections, checking along the way
    for round in 1:20, _ in 1:500
        d = SVector(rand(rng), rand(rng))
        QD.add!(a, [0.0], Evaluation(-100 * rand(rng), d))
        round % 4 == 0 && QD.add!(a, [0.0], Evaluation(-100 * rand(rng) + 50, d))
    end
    check(a)
    @test length(a) > 100

    # a rejected solution changes nothing
    before = (qd_score(a), max_fitness(a), mean_fitness(a), length(a))
    elite_descriptor = first(QD.elites(a))[2].descriptor
    @test QD.add!(a, [0.0], Evaluation(-1e6, elite_descriptor)).status == QD.Rejected
    @test (qd_score(a), max_fitness(a), mean_fitness(a), length(a)) == before

    # a solution that beats the maximum raises it, and one that does not leaves it
    best = max_fitness(a)
    QD.add!(a, [0.0], Evaluation(best + 5, SVector(0.5, 0.5)))
    @test max_fitness(a) == best + 5
    check(a)

    # empty! resets the totals
    empty!(a)
    @test qd_score(a) == 0
    @test coverage(a) == 0
    @test_throws "max_fitness is undefined for an archive with no elites" max_fitness(a)
    @test_throws "mean_fitness is undefined for an archive with no elites" mean_fitness(a)
    QD.add!(a, [0.0], Evaluation(-7.0, SVector(0.1, 0.1)))
    @test max_fitness(a) == -7.0     # not the maximum from before empty!
    @test mean_fitness(a) == -7.0
    @test qd_score(a; offset=10) == 3.0

    # negative fitness: the maximum is the largest, not the one of smallest magnitude
    b = GridArchive((4,), ((0, 1),))
    QD.add!(b, [0.0], Evaluation(-5.0, SVector(0.1)))
    QD.add!(b, [0.0], Evaluation(-2.0, SVector(0.6)))
    QD.add!(b, [0.0], Evaluation(-9.0, SVector(0.9)))
    @test max_fitness(b) == -2.0
end

@testset "GridArchive running sum rounding" begin
    # One million improvements spread over 10 000 cells. The exact sum is taken
    # in BigFloat from the stored fitness values.
    function sum_error(F; updates=1_000_000)
        rng = Xoshiro(5)
        a = GridArchive{Vector{Float64},F}((100, 100), ((0, 1), (0, 1)))
        centers = [SVector((i - 0.5) / 100, (j - 0.5) / 100) for i in 1:100 for j in 1:100]
        naive = 0.0
        stored = Dict{Int,F}()
        for _ in 1:updates
            c = rand(rng, 1:10_000)
            old = get(stored, c, nothing)
            new = old === nothing ? F(-130 - 100 * rand(rng)) : F(old + 0.1 * rand(rng))
            r = QD.add!(a, [0.0], Evaluation(new, centers[c]))
            if r.status == QD.NewCell
                naive += Float64(new)
            elseif r.status == QD.Improved
                naive += Float64(new) - Float64(old)
            end
            r.status == QD.Rejected || (stored[c] = new)
        end
        exact = sum(BigFloat(f) for f in values(stored))
        relative(x) = Float64(abs(BigFloat(x) - exact) / abs(exact))
        return relative(mean_fitness(a) * length(a)), relative(naive)
    end

    running, naive = sum_error(Float64)
    @test running <= 4 * eps()
    @test naive > 100 * running    # a plain running sum is much less accurate
    running32, _ = sum_error(Float32)
    @test running32 <= 4 * eps()
end
