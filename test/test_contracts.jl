@testset "evaluation types" begin
    ev = Evaluation(1.5, (0.2, 0.7))
    @test ev isa Evaluation{Float64,Tuple{Float64,Float64},Nothing}
    @test ev.phenotype === nothing
    @test Evaluation{Float64,Tuple{Float64,Float64}}(1.5, (0.2, 0.7)) === ev
    labelled = Evaluation(1.5, (0.2, 0.7), :ring)
    @test labelled isa Evaluation{Float64,Tuple{Float64,Float64},Symbol}
    @test labelled.phenotype === :ring
    r = AddResult(QD.NewCell, 0.5)
    @test r.status == QD.NewCell
    @test r.improvement === 0.5
    @test Int(QD.Rejected) < Int(QD.Improved) < Int(QD.NewCell)
end

@testset "role types" begin
    @test isabstracttype(AbstractQDProblem)
    @test isabstracttype(AbstractArchive)
    @test isabstracttype(AbstractVariation)
    @test isabstracttype(AbstractEmitter)
    @test AbstractArchive{Int,Float64,Float64,Nothing} isa Type
end

const INTERFACE_FUNCTIONS = (:evaluate, :evaluate_batch, :random_solution,
                             :descriptor_bounds, :descriptor_categories, :fitness_offset, :add!, :add_batch!,
                             :cell_index, :sample_elites, :elites, :capacity, :vary,
                             :ask, :tell!, :batch_size)

# Base.Docs.hasdoc and Base.ispublic are not available on Julia 1.10.
function hasdoc(name::Symbol)
    value = getfield(QD, name)
    # a module's docstring is stored in the module itself
    value isa Module && return haskey(Base.Docs.meta(value), Base.Docs.Binding(value, name))
    return haskey(Base.Docs.meta(QD), Base.Docs.Binding(QD, name))
end

@testset "interface functions are public, not exported" begin
    for f in INTERFACE_FUNCTIONS
        VERSION >= v"1.11" && @test Base.ispublic(QD, f)
        @test !Base.isexported(QD, f)
        @test hasdoc(f)
    end
    for f in (:solve, :init, :step!, :solve!, :qd_score, :coverage, :max_fitness, :mean_fitness)
        @test Base.isexported(QD, f)
    end
end

@testset "the docstrings of the algorithms are on the algorithms" begin
    for (name, phrase) in ((:MOME, "Pierrot"), (:MAPElites, "MAP-Elites"), (:CMAME, "Fontaine"), (:MELS, "Mace"),
                           (:DDS, "Lee"), (:CMAMAE, "Nikolaidis"))
        multidoc = Base.Docs.meta(QD)[Base.Docs.Binding(QD, name)]
        @test any(sig -> occursin(phrase, string(multidoc.docs[sig].text...)), multidoc.order)
    end
end

@testset "every public name is documented" begin
    for name in names(QD)
        name === :QualityDiversity && continue
        getfield(QD, name) isa AddStatus && continue  # documented on AddStatus
        name in (:solve, :init, :step!, :solve!) && continue  # documented where defined
        @test hasdoc(name)
    end
end

@testset "QDAlgorithm" begin
    struct DummyArchive <: AbstractArchive{Int,Float64,Float64,Nothing} end
    struct DummyEmitter <: AbstractEmitter end
    alg = QDAlgorithm(DummyArchive(), (DummyEmitter(), DummyEmitter()))
    @test alg isa QDAlgorithm{DummyArchive,Tuple{DummyEmitter,DummyEmitter}}
    @test_throws MethodError QDAlgorithm(DummyArchive(), [DummyEmitter()])
    @test QD.tell!(DummyEmitter(), [], [], []) === nothing
end

@testset "the constructors share their keywords and defaults" begin
    prob = Rastrigin(4)
    labels = QDProblem(x -> (0.0, (1, :a)); init=rng -> rand(rng, 2), categories=(1:2, [:a]))
    noisy = StochasticQDProblem((rng, x) -> (-sum(abs2, x), (x[1], x[2]) .+ 0.05 .* randn(rng, 2));
                                init=rng -> randn(rng, 3), bounds=((-3, 3), (-3, 3)))
    spread = SampledSpread(noisy; num_samples=2)
    grid_algorithms = (p -> MAPElites(p; dims=(2, 2)), p -> CMAME(p; dims=(2, 2), sigma0=0.5),
                       p -> CMAMAE(p; dims=(2, 2), sigma0=0.5, threshold_min=-100),
                       p -> DDS(p; dims=(2, 2), sigma0=0.5, bandwidth=0.5),
                       p -> CMAMEGA(p; dims=(2, 2), sigma0=1.0),
                       p -> CMAMAEGA(p; dims=(2, 2), sigma0=1.0, threshold_min=-100),
                       p -> OMGMEGA(p; dims=(2, 2), sigma0=1.0))
    grid(alg) = alg.archive isa GridArchive ? alg.archive : alg.archive.inner

    # every grid-based algorithm clamps a descriptor outside the bounds by default, takes
    # bounds=nothing for the bounds of the problem and fitness as any Real type
    for make in grid_algorithms
        alg = make(prob)
        @test grid(alg).outside === ClampOutside()
        @test QD.descriptor_bounds(grid(alg)) == QD.descriptor_bounds(prob)
        @test grid(make(prob)) isa GridArchive{Vector{Float64},Float64}
    end
    @test MOME(BiSphere(4); dims=(2, 2)).archive.outside === ClampOutside()
    @test MELS(spread; dims=(2, 2)).archive.inner.outside === ClampOutside()
    @test MAPElites(prob; dims=(2, 2), fitness=Int).archive isa GridArchive{Vector{Float64},Int}
    @test CMAME(prob; dims=(2, 2), sigma0=0.5, fitness=Float32).archive isa GridArchive{Vector{Float64},Float32}
    @test_throws "a ThresholdArchive needs a floating-point fitness type such as Float64, got Int64" CMAMAE(prob; dims=(2, 2), sigma0=0.5, threshold_min=-100, fitness=Int)
    @test_throws "a ThresholdArchive needs a floating-point fitness type" CMAMAEGA(prob; dims=(2, 2), sigma0=1.0, threshold_min=-100, fitness=Int)
    @test_throws "a DensityArchive needs a floating-point fitness type" DDS(prob; dims=(2, 2), sigma0=0.5, bandwidth=0.5, fitness=Int)

    # a categorical problem gets a CategoricalArchive from MAPElites, which cannot clamp,
    # and the error that says so from the others instead of a MethodError
    @test MAPElites(labels).archive isa CategoricalArchive
    @test MAPElites(labels).archive.outside === ThrowOutside()
    for make in grid_algorithms[2:end]
        @test_throws "has descriptor_categories, not descriptor_bounds" make(labels)
    end
    @test_throws "has descriptor_categories, not descriptor_bounds" MOME(labels; dims=(2, 2), reference_point=[0.0, 0.0])
    for make in (p -> GridArchive(p; dims=(2, 2)), p -> CVTArchive(p; cells=3, samples=10),
                 p -> SlidingBoundariesArchive(p; dims=(2, 2)), p -> DeviceGridArchive(p; dims=(2, 2), device=Array),
                 p -> DeviceCVTArchive(p; cells=3, samples=10, device=Array),
                 p -> ParetoGridArchive(p; dims=(2, 2), reference_point=[0.0, 0.0]))
        @test_throws "has descriptor_categories, not descriptor_bounds" make(labels)
        @test QD.descriptor_bounds(make(prob)) == QD.descriptor_bounds(prob)
    end
    struct NoBounds <: AbstractQDProblem end
    @test_throws "NoBounds does not define descriptor_bounds" QD.problem_bounds(NoBounds(), nothing)
    @test QD.problem_bounds(NoBounds(), ((0, 1),)) == ((0, 1),)

    # a problem whose random solutions leave the bounds no longer fails on the first step
    wide = QDProblem(x -> (-sum(abs2, x), (x[1], x[2])); init=rng -> 4 .* randn(rng, 2), bounds=((-1, 1), (-1, 1)))
    @test length(init(wide, MAPElites(wide; dims=(4, 4)); budget=1_000, rng=Xoshiro(1)).archive) > 0
    @test_throws "outside the archive bounds" init(wide, MAPElites(wide; dims=(4, 4), outside=ThrowOutside()); budget=1_000, rng=Xoshiro(1))
    @test_throws "outside the archive bounds" init(wide, MAPElites(GridArchive(wide; dims=(4, 4))); budget=1_000, rng=Xoshiro(1))

    # the raw archives still throw by default
    for archive in (GridArchive(prob; dims=(2, 2)), CVTArchive(prob; cells=3, samples=10),
                    ParetoGridArchive(BiSphere(4); dims=(2, 2), reference_point=[0.0, 0.0]),
                    DeviceGridArchive(prob; dims=(2, 2), device=Array))
        @test archive.outside === ThrowOutside()
    end
end
