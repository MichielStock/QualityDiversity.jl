using Aqua: Aqua
using ExplicitImports: ExplicitImports
using JET: JET

@testset "Aqua" begin
    Aqua.test_all(QualityDiversity)
end

# These names are used on purpose although their owners do not declare them
# public: the CommonSolve verbs are the interface this package extends, and the
# others are stable, documented Base and Random functions (`require_one_based_indexing`
# is declared public from Julia 1.11 on).
const NON_PUBLIC_ON_PURPOSE = (:init, :solve, :solve!, :step!, :default_rng, :promote_op, :tail,
                               :require_one_based_indexing)

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(QualityDiversity) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(QualityDiversity) === nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(QualityDiversity) === nothing
    @test ExplicitImports.check_all_qualified_accesses_via_owners(QualityDiversity) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(QualityDiversity) === nothing
    @test ExplicitImports.check_all_explicit_imports_are_public(QualityDiversity; ignore=NON_PUBLIC_ON_PURPOSE) === nothing
    @test ExplicitImports.check_all_qualified_accesses_are_public(QualityDiversity; ignore=NON_PUBLIC_ON_PURPOSE) === nothing
end

@testset "JET" begin
    JET.test_package(QualityDiversity; target_modules=(QualityDiversity,))

    prob = Rastrigin(4)
    alg = MAPElites(prob; dims=(5, 5), batch_size=8)
    state = init(prob, alg; budget=64, rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(state)
    JET.@test_opt target_modules = (QualityDiversity,) QD.add!(state.archive, [0.0, 0.0, 0.0, 0.0],
                                                            Evaluation(1.0, SVector(0.0, 0.0)))
    JET.@test_opt target_modules = (QualityDiversity,) QD.cell_index(state.archive, SVector(0.0, 0.0))
    JET.@test_opt target_modules = (QualityDiversity,) QD.sample_elites(Xoshiro(1), state.archive, 4)
    JET.@test_opt target_modules = (QualityDiversity,) QD.ask(Xoshiro(1), state.emitters[1], state.archive)
    JET.@test_opt target_modules = (QualityDiversity,) QD.vary(Xoshiro(1), GaussianMutation(0.1), [0.0, 0.0])
end
