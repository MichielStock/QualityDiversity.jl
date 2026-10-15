using Makie: Makie, Heatmap, Figure, Axis, heatmap, heatmap!

const MakieExt = Base.get_extension(QualityDiversity, :QualityDiversityMakieExt)

@testset "extension loads" begin
    @test MakieExt isa Module
end

convert_heatmap(args...; kwargs...) = Makie.convert_arguments(Heatmap, args...; kwargs...)

@testset "GridArchive" begin
    archive = GridArchive((3, 2), ((0, 3), (10, 12)))
    QD.add!(archive, [0.0], Evaluation(1.5, SVector(0.5, 10.5)))   # cell (1, 1)
    QD.add!(archive, [0.0], Evaluation(-2.0, SVector(2.5, 11.5)))  # cell (3, 2)
    xs, ys, values = convert_heatmap(archive)
    @test xs == [0.0, 1.0, 2.0, 3.0]
    @test ys == [10.0, 11.0, 12.0]
    @test size(values) == (3, 2)
    @test values[1, 1] == 1.5
    @test values[3, 2] == -2.0
    @test count(isnan, values) == 4
    @test Makie.plottype(archive) === Heatmap

    @test all(isnan, convert_heatmap(GridArchive((2, 2), ((0, 1), (0, 1))))[3])
end

@testset "fitness types are shown as Float64" begin
    archive = GridArchive{Vector{Float64},Float32}((2, 2), ((0, 1), (0, 1)))
    QD.add!(archive, [0.0], Evaluation(0.25f0, SVector(0.1, 0.1)))
    values = convert_heatmap(archive)[3]
    @test eltype(values) === Float64
    @test values[1, 1] == 0.25
end

@testset "CVTArchive" begin
    centroids = [SVector(0.25, 0.5), SVector(0.75, 0.5)]
    archive = CVTArchive(centroids, ((0, 1), (0, 1)))
    QD.add!(archive, [0.0], Evaluation(2.0, SVector(0.1, 0.5)))
    xs, ys, values = convert_heatmap(archive; pixels=4)
    @test xs == [0.0, 0.25, 0.5, 0.75, 1.0]
    @test ys == xs
    @test size(values) == (4, 4)
    @test all(==(2.0), values[1:2, :])
    @test all(isnan, values[3:4, :])
    @test size(convert_heatmap(archive)[3]) == (200, 200)
    @test_throws "pixels must be at least 1, got 0" convert_heatmap(archive; pixels=0)
end

@testset "ParetoGridArchive" begin
    archive = ParetoGridArchive{Vector{Float64}}((2, 2), ((0, 1), (0, 1)), [0.0, 0.0])
    QD.add!(archive, [0.0], Evaluation(SVector(1.0, 3.0), SVector(0.2, 0.7)))   # cell (1, 2)
    QD.add!(archive, [0.0], Evaluation(SVector(3.0, 1.0), SVector(0.2, 0.7)))
    _, _, values = convert_heatmap(archive)
    @test values[1, 2] == 5.0
    @test count(isnan, values) == 3
end

@testset "ThresholdArchive shows the result archive" begin
    archive = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1))); learning_rate=0.5, threshold_min=-10)
    QD.add!(archive, [0.0], Evaluation(1.0, SVector(0.1, 0.1)))
    QD.add!(archive, [0.0], Evaluation(3.0, SVector(0.1, 0.1)))
    @test isequal(convert_heatmap(archive)[3], convert_heatmap(QD.result_archive(archive))[3])
    @test convert_heatmap(archive)[3][1, 1] == 3.0

    cvt = ThresholdArchive(CVTArchive([SVector(0.25, 0.5), SVector(0.75, 0.5)], ((0, 1), (0, 1)));
                           learning_rate=0.5, threshold_min=-10)
    @test size(convert_heatmap(cvt; pixels=8)[3]) == (8, 8)
end

@testset "SpreadArchive shows the archive it wraps" begin
    archive = SpreadArchive(GridArchive((2, 2), ((0, 1), (0, 1))))
    QD.add!(archive, [0.0], Evaluation(2.0, QD.DescriptorSamples([SVector(0.1, 0.1), SVector(0.2, 0.1)])))
    @test isequal(convert_heatmap(archive)[3], convert_heatmap(archive.inner)[3])
    @test convert_heatmap(archive)[3][1, 1] == 2.0
    cvt = SpreadArchive(CVTArchive([SVector(0.25, 0.5), SVector(0.75, 0.5)], ((0, 1), (0, 1))))
    @test size(convert_heatmap(cvt; pixels=8)[3]) == (8, 8)
end

@testset "other dimensions throw" begin
    @test_throws "a heatmap needs an archive with 2 descriptor dimensions but this GridArchive has 3" convert_heatmap(
        GridArchive((2, 2, 2), ((0, 1), (0, 1), (0, 1))))
    @test_throws "but this CVTArchive has 1" convert_heatmap(
        CVTArchive([SVector(0.25), SVector(0.75)], ((0, 1),)))
end

@testset "plotting a run" begin
    prob = Rastrigin(4)
    result = solve(prob, MAPElites(prob; dims=(8, 8), batch_size=16); budget=400, rng=Xoshiro(1))
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot = heatmap!(ax, result.archive)
    @test plot isa Heatmap
    @test Makie.plot(result.archive) isa Makie.FigureAxisPlot
end
