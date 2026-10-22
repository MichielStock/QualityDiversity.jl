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
    # an emitter with state keeps the batch it asked for, so one instance cannot run twice
    cma = CMAEmitter(2; sigma0=0.5)
    @test_throws "emitters[1] and emitters[3] are the same CMAEmitter; every emitter with state must be a separate instance" QDAlgorithm(DummyArchive(), (cma, DummyEmitter(), cma))
    @test QDAlgorithm(DummyArchive(), (cma, CMAEmitter(2; sigma0=0.5))) isa QDAlgorithm
    @test QDAlgorithm(DummyArchive(), Tuple(CMAEmitter(2; sigma0=0.5) for _ in 1:3)).emitters isa NTuple{3,CMAEmitter}
end
