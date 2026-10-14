@testset "evaluation types" begin
    ev = Evaluation(1.5, (0.2, 0.7))
    @test ev isa Evaluation{Float64,Tuple{Float64,Float64}}
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
    @test AbstractArchive{Int,Float64,Float64} isa Type
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

@testset "every public name is documented" begin
    for name in names(QD)
        name === :QualityDiversity && continue
        getfield(QD, name) isa AddStatus && continue  # documented on AddStatus
        name in (:solve, :init, :step!, :solve!) && continue  # documented where defined
        @test hasdoc(name)
    end
end

@testset "QDAlgorithm" begin
    struct DummyArchive <: AbstractArchive{Int,Float64,Float64} end
    struct DummyEmitter <: AbstractEmitter end
    alg = QDAlgorithm(DummyArchive(), (DummyEmitter(), DummyEmitter()))
    @test alg isa QDAlgorithm{DummyArchive,Tuple{DummyEmitter,DummyEmitter}}
    @test_throws MethodError QDAlgorithm(DummyArchive(), [DummyEmitter()])
    @test QD.tell!(DummyEmitter(), [], [], []) === nothing
end
