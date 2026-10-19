include("reference/density.jl")

φ(z) = exp(-z^2 / 2) / sqrt(2π)
line_grid() = GridArchive((4,), ((0, 1),); outside=ClampOutside())

@testset "DensityArchive construction" begin
    a = DensityArchive(Xoshiro(1), GridArchive((10, 10), ((0, 1), (0, 1))); bandwidth=0.05)
    @test a isa AbstractArchive{Vector{Float64},Float64,SVector{2,Float64},Nothing}
    @test length(a) == 0
    @test QD.capacity(a) == 100
    @test length(a.buffer) == 10_000
    @test QD.result_archive(a) === a.inner
    @test QD.descriptor_bounds(a) == ((0.0, 1.0), (0.0, 1.0))
    rng = Xoshiro(7)
    DensityArchive(rng, line_grid(); bandwidth=1)
    @test rng == Xoshiro(7)                     # the archive draws from its own copy

    @test_throws "bandwidth must be finite and greater than 0, got 0" DensityArchive(Xoshiro(1), line_grid(); bandwidth=0)
    @test_throws "bandwidth must be finite and greater than 0, got Inf" DensityArchive(Xoshiro(1), line_grid(); bandwidth=Inf)
    @test_throws "buffer_size must be at least 1, got 0" DensityArchive(Xoshiro(1), line_grid(); bandwidth=1, buffer_size=0)
    full = line_grid()
    QD.add!(full, [1.0], Evaluation(1.0, SVector(0.5)))
    @test_throws "the archive to wrap must be empty but holds 1 solutions" DensityArchive(Xoshiro(1), full; bandwidth=1)
    @test_throws "wraps an archive with SVector descriptors" DensityArchive(Xoshiro(1), CategoricalArchive((1:3,)); bandwidth=1)
    @test_throws "wraps an archive with SVector descriptors" DensityArchive(Xoshiro(1), GridArchive{Vector{Float64},Int}((2,), ((0, 1),)); bandwidth=1)
end

@testset "DensityArchive density" begin
    a = DensityArchive(Xoshiro(1), line_grid(); bandwidth=0.5)
    @test QD.density(a, SVector(0.3)) == 0
    @test QD.add!(a, [1.0], Evaluation(1.0, SVector(0.0))) == AddResult(QD.NewCell, -0.0)
    # the density is that of the buffer before the solution enters it
    @test QD.add!(a, [2.0], Evaluation(2.0, SVector(1.0))) == AddResult(QD.NewCell, -φ(2.0) / 0.5)
    @test QD.density(a, SVector(0.5)) ≈ (φ(1.0) + φ(1.0)) / (2 * 0.5)
    @test QD.add!(a, [0.5], Evaluation(0.5, SVector(0.1))).status == QD.Rejected    # the grid keeps the better one
    @test QD.result_archive(a) === a.inner
    @test length(a) == 2

    @test_throws "descriptor component 1 is NaN, which is not finite" QD.add!(a, [0.0], Evaluation(1.0, SVector(NaN)))
    @test_throws "descriptor has 2 dimensions but the archive has 1" QD.density(a, SVector(0.0, 0.0))
    @test a.filled == 3

    # every result of a batch is against the buffer before the batch
    b = DensityArchive(Xoshiro(1), line_grid(); bandwidth=0.5)
    rs = QD.add_batch!(b, [[1.0], [2.0]], [Evaluation(1.0, SVector(0.0)), Evaluation(2.0, SVector(0.9))])
    @test [r.improvement for r in rs] == [0.0, 0.0]
    @test [r.status for r in rs] == [QD.NewCell, QD.NewCell]
    @test b.filled == 2

    empty!(b)
    @test length(b) == 0
    @test b.filled == 0
    @test QD.density(b, SVector(0.0)) == 0
end

@testset "DensityArchive matches pyribs" begin
    # reference/density.jl: the densities pyribs reports for a sequence of batches
    for case in density_cases
        N = case.dims
        total = length(case.descriptors) ÷ N
        descriptors = [SVector{N}(case.descriptors[(i-1)*N+1:i*N]) for i in 1:total]
        grid = GridArchive(ntuple(_ -> 4, N), ntuple(_ -> (-1, 1), N); outside=ClampOutside())
        a = DensityArchive(Xoshiro(1), grid; bandwidth=case.bandwidth, buffer_size=total)
        densities = Float64[]
        for b in 1:total÷case.batch
            ids = (b-1)*case.batch+1:b*case.batch
            rs = QD.add_batch!(a, [[Float64(i)] for i in ids], [Evaluation(0.0, descriptors[i]) for i in ids])
            append!(densities, [-r.improvement for r in rs])
        end
        @test densities ≈ case.densities rtol = 1e-12
    end
end

@testset "DensityArchive reservoir" begin
    # after a stream of 20_000 descriptors, every one is in a buffer of 100 with the same
    # probability, 1/200: the stream's first and last quarters are equally represented
    counts = zeros(Int, 4)
    for seed in 1:200
        a = DensityArchive(Xoshiro(seed), GridArchive((2,), ((0, 20_000),)); bandwidth=1, buffer_size=100)
        for i in 1:20_000
            QD.add!(a, [0.0], Evaluation(0.0, SVector(Float64(i))))
        end
        @test a.filled == 100
        for d in a.buffer
            counts[clamp(ceil(Int, d[1] / 5_000), 1, 4)] += 1
        end
    end
    # 20_000 entries in all; each quarter expects 5_000, with a standard deviation near 61
    @test sum(counts) == 20_000
    @test all(c -> abs(c - 5_000) < 300, counts)

    # the same holds when batches replace several entries at once: 10_000 descriptors in
    # batches of 100 into a buffer of 50, each quarter expecting 1_250 of 5_000 entries,
    # with a standard deviation near 31
    batched = zeros(Int, 4)
    for seed in 1:100
        a = DensityArchive(Xoshiro(seed), GridArchive((2,), ((0, 10_000),)); bandwidth=1, buffer_size=50)
        for b in 1:100
            ids = (b-1)*100+1:b*100
            QD.add_batch!(a, [[0.0] for _ in ids], [Evaluation(0.0, SVector(Float64(i))) for i in ids])
        end
        for d in a.buffer
            batched[clamp(ceil(Int, d[1] / 2_500), 1, 4)] += 1
        end
    end
    @test sum(batched) == 5_000
    @test all(c -> abs(c - 1_250) < 150, batched)

    # the same seed gives the same buffer
    fill_buffer(seed) = (a = DensityArchive(Xoshiro(seed), line_grid(); bandwidth=1, buffer_size=10);
                         QD.add_batch!(a, [[0.0] for _ in 1:500], [Evaluation(0.0, SVector(i / 500)) for i in 1:500]); a.buffer)
    @test fill_buffer(3) == fill_buffer(3)
    @test fill_buffer(3) != fill_buffer(4)
end

@testset "DDS" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1]^3, x[2]^3)); init=rng -> 0.1 .* randn(rng, 6),
                     bounds=((-8, 8), (-8, 8)))
    alg = DDS(Xoshiro(1), prob; dims=(10, 10), sigma0=0.3, bandwidth=0.8, num_emitters=3, batch_size=12)
    @test alg.archive isa DensityArchive
    @test length(alg.emitters) == 3
    @test QD.batch_size(alg.emitters[1]) == 12
    result = solve(prob, alg; budget=4_000, rng=Xoshiro(2))
    @test result.archive isa GridArchive
    again = solve(prob, alg; budget=4_000, rng=Xoshiro(2))
    @test collect(QD.elites(again.archive)) == collect(QD.elites(result.archive))
    @test alg.archive.filled == 0                     # init works on a copy
    # it spreads out, while CMA-ME with the same emitters stays near its start
    cmame = solve(prob, CMAME(prob; dims=(10, 10), sigma0=0.3, num_emitters=3, batch_size=12, outside=ClampOutside());
                  budget=4_000, rng=Xoshiro(2))
    @test coverage(result.archive) > coverage(cmame.archive)

    @test_throws "dims" DDS(Xoshiro(1), prob; sigma0=0.3, bandwidth=0.8)
end
