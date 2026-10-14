@testset "CategoricalArchive construction" begin
    a = CategoricalArchive((0:3, [20, 10]))
    @test a isa CategoricalArchive{Vector{Float64},Float64,2}
    @test a isa QD.CellArchive{Vector{Float64},Float64,SVector{2,Float64}}
    @test QD.capacity(a) == 8 && length(a) == 0 && coverage(a) == 0
    @test a.categories == ([0.0, 1.0, 2.0, 3.0], [10.0, 20.0])          # sorted
    @test CategoricalArchive{Vector{Float32},Float32}(([1, 2],)) isa
          CategoricalArchive{Vector{Float32},Float32,1}
    @test CategoricalArchive(([0.5, 1.5],); outside=RejectOutside()).outside isa RejectOutside

    @test_throws "categories must have at least one entry" CategoricalArchive(())
    @test_throws "categories[2] is empty" CategoricalArchive((1:3, Int[]))
    @test_throws "categories[1] contains 2.0 more than once" CategoricalArchive(([2, 1, 2],))
    @test_throws "categories[1] is [1.0, NaN] but its values must be finite" CategoricalArchive(([1, NaN],))
    @test_throws "categories[1] is [1.0, Inf] but its values must be finite" CategoricalArchive(([1, Inf],))
    @test_throws "ClampOutside() has no meaning for a CategoricalArchive" CategoricalArchive((1:3,); outside=ClampOutside())
end

@testset "CategoricalArchive cell_index" begin
    a = CategoricalArchive((0:3, [10, 20], [-1.5, 0.5, 2.5]))
    @test QD.capacity(a) == 24
    # cells are numbered like a grid: ascending values, first dimension fastest
    expected = 0
    for k in 1:3, j in 1:2, i in 1:4
        expected += 1
        @test QD.cell_index(a, [i - 1, 10 * j, [-1.5, 0.5, 2.5][k]]) == expected
    end
    @test QD.cell_index(a, SVector(2.0, 20.0, 0.5)) == QD.cell_index(a, [2, 20, 0.5])

    @test_throws "descriptor has 2 dimensions but the archive has 3" QD.cell_index(a, [0, 10])
    @test_throws "descriptor component 1 is 1.5, which is not one of the categories [0.0, 1.0, 2.0, 3.0]" QD.cell_index(a, [1.5, 10, 0.5])
    @test_throws "descriptor component 2 is 15.0, which is not one of the categories [10.0, 20.0]" QD.cell_index(a, [1, 15, 0.5])
    @test_throws "descriptor component 3 is NaN, which is not one of the categories" QD.cell_index(a, [1, 10, NaN])
    @test_throws "descriptor component 1 is 4.0" QD.add!(a, [1.0], Evaluation(1.0, SVector(4.0, 10.0, 0.5)))
    @test_throws "(100 values)" QD.cell_index(CategoricalArchive((1:100,)), [0.5])
    @test_throws "[1.0, 2.0, 3.0, 4.0] … [99.0, 100.0] (100 values)" QD.cell_index(CategoricalArchive((1:100,)), [0.5])

    rejecting = CategoricalArchive((0:3, [10, 20]); outside=RejectOutside())
    @test QD.cell_index(rejecting, [1.5, 10]) == 0
    @test QD.cell_index(rejecting, [1, NaN]) == 0
    @test QD.cell_index(rejecting, [1, 10]) == 2
    r = QD.add!(rejecting, [1.0], Evaluation(1.0f0, SVector(1.5, 10.0)))
    @test r.status == QD.Rejected && r.improvement === 0.0 && length(rejecting) == 0
end

@testset "CategoricalArchive stores and reports like any cell archive" begin
    rng = Xoshiro(2)
    a = CategoricalArchive((0:9, 0:9))
    ev(f, d) = Evaluation(f, SVector{2,Float64}(d))
    @test QD.add!(a, [1.0], ev(-3.0, (2, 3))).status == QD.NewCell
    @test QD.add!(a, [2.0], ev(-1.0, (2, 3))).status == QD.Improved
    @test QD.add!(a, [3.0], ev(-2.0, (2, 3))).status == QD.Rejected
    @test QD.add!(a, [4.0], ev(-9.0, (3, 2))).status == QD.NewCell      # not the same cell as (2, 3)
    @test sort(first.(collect(QD.elites(a)))) == [[2.0], [4.0]]

    for _ in 1:3000
        QD.add!(a, [0.0], ev(-100 * rand(rng), (rand(rng, 0:9), rand(rng, 0:9))))
    end
    @test length(a) == 100 && coverage(a) == 1.0
    @test qd_score(a; offset=100) ≈ generic_qd_score(a; offset=100) rtol = 1e-14
    @test max_fitness(a) == generic_max_fitness(a)
    @test mean_fitness(a) ≈ generic_mean_fitness(a) rtol = 1e-14
    empty!(a)
    @test length(a) == 0 && qd_score(a) == 0
    @test_throws "max_fitness is undefined" max_fitness(a)
end

@testset "MAPElites over a CategoricalArchive" begin
    # the descriptor is the rounded coordinates, which mutation can push outside [-3, 3]
    prob = QDProblem(x -> (-sum(abs2, x), (round(clamp(x[1], -3, 3)), round(clamp(x[2], -3, 3))));
                     init=rng -> 6 .* rand(rng, 2) .- 3, bounds=((-3, 3), (-3, 3)))
    archive = CategoricalArchive((-3:3, -3:3))
    alg = MAPElites(archive; batch_size=20, variation=GaussianMutation(1.0))
    result = solve(prob, alg; budget=2_000, rng=Xoshiro(1))
    @test result.n_evals == 2_000
    @test length(result.archive) == 49 && coverage(result.archive) == 1.0
    @test length(archive) == 0
    @test solve(prob, alg; budget=2_000, rng=Xoshiro(1)).history == result.history
    # the best solution of a cell is at least as good as the cell centre's neighbours
    @test max_fitness(result.archive) > -0.1

    wrong = CategoricalArchive((1:3,))
    @test_throws "the problem has 2 descriptor dimensions but the archive has 1" init(prob, MAPElites(wrong); budget=100)
end
