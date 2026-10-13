using QualityDiversity
using Test

const QD = QualityDiversity

@testset "QualityDiversity" begin
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

    @testset "interface functions are public, not exported" begin
        for f in (:evaluate, :evaluate_batch, :random_solution, :descriptor_bounds,
                  :fitness_offset, :add!, :add_batch!, :cell_index, :sample_elites,
                  :elites, :capacity, :vary, :ask, :tell!, :batch_size)
            @test Base.ispublic(QD, f)
            @test !Base.isexported(QD, f)
            @test isempty(methods(getfield(QD, f)))
        end
        for f in (:solve, :qd_score, :coverage, :max_fitness, :mean_fitness)
            @test Base.isexported(QD, f)
        end
    end

    @testset "every public name is documented" begin
        for name in names(QD; all=false)
            name === :QualityDiversity && continue
            getfield(QD, name) isa AddStatus && continue  # documented on AddStatus
            name === :solve && continue                   # documented in CommonSolve
            @test Base.Docs.hasdoc(QD, name)
        end
        for f in (:evaluate, :evaluate_batch, :random_solution, :descriptor_bounds,
                  :fitness_offset, :add!, :add_batch!, :cell_index, :sample_elites,
                  :elites, :capacity, :vary, :ask, :tell!, :batch_size)
            @test Base.Docs.hasdoc(QD, f)
        end
    end

    @testset "QDAlgorithm" begin
        struct DummyArchive <: AbstractArchive{Int,Float64,Float64} end
        struct DummyEmitter <: AbstractEmitter end
        alg = QDAlgorithm(DummyArchive(), (DummyEmitter(), DummyEmitter()))
        @test alg isa QDAlgorithm{DummyArchive,Tuple{DummyEmitter,DummyEmitter}}
        @test_throws MethodError QDAlgorithm(DummyArchive(), [DummyEmitter()])
    end
end
