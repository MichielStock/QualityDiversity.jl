using Makie: Makie, ECDFPlot, Heatmap, Lines, Figure, Axis, heatmap, heatmap!, ecdfplot, lines

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

# A category value that has no order.
struct Unsorted end

# The category labels of the axes of `ax`, from left to right and bottom to top.
function category_labels(ax)
    labels(conversion) = Makie.get_ticks(conversion, Makie.automatic, identity, Makie.automatic, 0, 10)[2]
    return labels(ax.dim1_conversion[]), labels(ax.dim2_conversion[])
end

@testset "CategoricalArchive" begin
    # categories in an order that is not sorted, so that the plot has to sort them
    archive = CategoricalArchive(([:ring, :alkane, :aromatic], [20, 10]))
    QD.add!(archive, [0.0], Evaluation(1.5, (:ring, 20)))
    QD.add!(archive, [0.0], Evaluation(-2.0, (:alkane, 10)))
    @test Makie.plottype(archive) === Heatmap
    xs, ys, values = convert_heatmap(archive)
    @test xs isa Makie.Categorical && ys isa Makie.Categorical
    @test xs.values == [:alkane, :aromatic, :ring]
    @test ys.values == [10, 20]
    @test size(values) == (3, 2)
    @test values[3, 2] == 1.5          # (:ring, 20)
    @test values[1, 1] == -2.0         # (:alkane, 10)
    @test count(isnan, values) == 4

    fig = Figure()
    ax = Axis(fig[1, 1])
    plot = heatmap!(ax, archive)
    @test plot isa Heatmap
    Makie.update_state_before_display!(fig)
    @test category_labels(ax) == (["alkane", "aromatic", "ring"], ["10", "20"])
    @test plot[1][] == [0.5, 1.5, 2.5, 3.5] && plot[2][] == [0.5, 1.5, 2.5]
    @test heatmap(archive) isa Makie.FigureAxisPlot

    @test_throws "but this CategoricalArchive has 3" convert_heatmap(CategoricalArchive((1:2, 1:2, 1:2)))
    @test_throws "categories[2] has values of type" convert_heatmap(CategoricalArchive((1:2, [Unsorted()])))
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

@testset "DensityArchive heatmap" begin
    archive = DensityArchive(Xoshiro(1), GridArchive((3, 2), ((0, 3), (10, 12))); bandwidth=0.5)
    QD.add!(archive, [0.0], Evaluation(1.5, SVector(0.5, 10.5)))
    @test Makie.plottype(archive) === Heatmap
    @test isequal(convert_heatmap(archive), convert_heatmap(QD.result_archive(archive)))
end

# The points of the stairs that Makie draws for an ECDF.
ecdf_points(x) = only(Makie.convert_arguments(ECDFPlot, x).args)

@testset "archive ECDF" begin
    archive = GridArchive((4,), ((0, 1),))
    for (f, d) in ((3.0, 0.1), (1.0, 0.4), (2.0, 0.9))
        QD.add!(archive, [0.0], Evaluation(f, SVector(d)))
    end
    # the ECDF of the elite fitness is that of the fitness values themselves
    @test ecdf_points(archive) == ecdf_points([3.0, 1.0, 2.0])
    @test ecdfplot(archive) isa Makie.FigureAxisPlot
    # a ThresholdArchive shows its result archive, whose elites can differ
    t = ThresholdArchive(GridArchive((4,), ((0, 1),)); learning_rate=0.5, threshold_min=0)
    QD.add!(t, [0.0], Evaluation(3.0, SVector(0.1)))
    QD.add!(t, [0.0], Evaluation(2.0, SVector(0.1)))        # passes the threshold of 1.5
    @test only(ev.fitness for (_, ev) in QD.elites(t)) == 2.0
    @test ecdf_points(t) == ecdf_points([3.0])
    d = DensityArchive(Xoshiro(1), GridArchive((4,), ((0, 1),)); bandwidth=0.5)
    QD.add!(d, [0.0], Evaluation(2.5, SVector(0.3)))
    @test ecdf_points(d) == ecdf_points([2.5])
    u = UnstructuredArchive(1; capacity=5, threshold=0.1)
    QD.add!(u, [0.0], Evaluation(-1.0, SVector(0.3)))
    @test ecdf_points(u) == ecdf_points([-1.0])
    f32 = GridArchive{Vector{Float64},Float32}((2,), ((0, 1),))
    QD.add!(f32, [0.0], Evaluation(0.25f0, SVector(0.1)))
    @test ecdf_points(f32) == ecdf_points([0.25])

    bi = BiSphere(2)
    p = ParetoGridArchive(bi; dims=(3, 3), reference_point=QD.reference_point(bi))
    @test_throws "an ECDF needs a single real fitness, but this ParetoGridArchive" ecdf_points(p)
end

@testset "CQD score plot" begin
    archive = GridArchive((4,), ((0, 1),))
    for (f, d) in ((3.0, 0.1), (1.0, 0.4), (2.0, 0.9))
        QD.add!(archive, [0.0], Evaluation(f, SVector(d)))
    end
    r = cqd_score(archive; targets=[[0.5]], fitness_range=(0, 3), penalties=3)
    @test Makie.plottype(r) === Lines
    xs, ys = Makie.convert_arguments(Lines, r)
    @test xs == [0.0, 0.5, 1.0]
    @test ys ≈ [1.0, 0.8, 0.6]
    @test lines(r) isa Makie.FigureAxisPlot
end
