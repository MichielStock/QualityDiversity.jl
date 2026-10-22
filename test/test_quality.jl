using Aqua: Aqua
using ExplicitImports: ExplicitImports

@testset "Aqua" begin
    Aqua.test_all(QualityDiversity)
end

# These names are used on purpose although their owners do not declare them
# public: the CommonSolve verbs are the interface this package extends, and the
# others are stable, documented Base, Random and Makie functions (`require_one_based_indexing`
# is declared public from Julia 1.11 on; `plottype` is how an extension tells Makie which
# plot `plot(x)` draws for its own type). `CellArchive`, `cell_index`, `result_archive`, `elites`, `evaluate`,
# `DeviceCellArchive`, `AddResultBatch` and the statuses `Rejected`, `Improved` and
# `NewCell` are declared public with `@compat public`, which ExplicitImports only sees from
# Julia 1.11 on; the extensions use them. The KernelAbstractions extension implements the device
# kernels behind internal functions of the package (`device_add_batch!` to
# `outside_improvement`), which are not part of its interface, and uses `@atomic` of Atomix,
# the documented atomic operations of KernelAbstractions kernels. The ForwardDiff extension
# implements `forwarddiff_evaluate_gradient` with `differentiable_values`, internal functions
# behind `evaluate_gradient`, and calls `jacobian`, the documented interface of ForwardDiff,
# which does not declare it public.
const NON_PUBLIC_ON_PURPOSE = (:init, :solve, :solve!, :step!, :default_rng, :promote_op, :tail,
                               :require_one_based_indexing, :plottype,
                               :CellArchive, :cell_index, :result_archive, :elites, :evaluate,
                               :DeviceCellArchive, :AddResultBatch, :Rejected, :Improved, :NewCell,
                               :device_add_batch!, :device_evaluate, :device_evaluate!, :device_vary!,
                               :refresh_occupied!, :solution_length, :fitness_gain, :outside_improvement, Symbol("@atomic"),
                               :forwarddiff_evaluate_gradient, :differentiable_values, :jacobian)

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(QualityDiversity) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(QualityDiversity) === nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(QualityDiversity) === nothing
    @test ExplicitImports.check_all_qualified_accesses_via_owners(QualityDiversity) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(QualityDiversity) === nothing
    @test ExplicitImports.check_all_explicit_imports_are_public(QualityDiversity; ignore=NON_PUBLIC_ON_PURPOSE) === nothing
    @test ExplicitImports.check_all_qualified_accesses_are_public(QualityDiversity; ignore=NON_PUBLIC_ON_PURPOSE) === nothing
end

# JET is pinned to minor versions of Julia (see the compat entry in Project.toml) and
# breaks on new Julia releases, which should not block the other CI jobs. Aqua and
# ExplicitImports always run; the JET tests run only when QD_JET is "1", which CI sets
# for one job. They live in their own file because `JET.@test_opt` is a macro: the
# block is expanded, and so needs JET loaded, before it runs.
if get(ENV, "QD_JET", "") == "1"
    include("test_jet.jl")
else
    @info "Skipping the JET tests: set QD_JET=1 to run them"
end
