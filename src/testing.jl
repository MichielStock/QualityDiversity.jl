"""
    QualityDiversity.Testing

Conformance checks for the types a user defines. Each function runs inside a
`@testset`, so failures are reported by the test runner of the caller:

```julia
using Test, QualityDiversity
@testset "MyProblem" begin
    QualityDiversity.Testing.test_problem(MyProblem())
end
```

The checks cover the contract of each role, whether the calls are inferrable,
that the results have the right types, and, for archives, that `cell_index`
(and `add!` for `isbits` genotypes) does not allocate.

- [`test_problem`](@ref QualityDiversity.Testing.test_problem)
- [`test_archive`](@ref QualityDiversity.Testing.test_archive)
- [`test_variation`](@ref QualityDiversity.Testing.test_variation)
- [`test_emitter`](@ref QualityDiversity.Testing.test_emitter)
"""
module Testing

using Random: AbstractRNG, Xoshiro
using Test: @inferred, @test, @testset

import ..QualityDiversity as QD

export test_problem, test_archive, test_variation, test_emitter

# Allocation of one call, measured after a warm-up call. The call goes through
# a function barrier so that the arguments have concrete types.
function allocations(f::F, args::Vararg{Any,N}) where {F,N}
    f(args...)
    return @allocated f(args...)
end

"""
    test_problem(prob; rng=Xoshiro(1), n=5, rtol=0)

Check that `prob` meets the contract of [`AbstractQDProblem`](@ref).

`random_solution` is called `n` times. Every solution must evaluate to an
[`Evaluation`](@ref) with a real or vector fitness and a descriptor that lies
inside `descriptor_bounds(prob)` and has one entry per bound. `evaluate` must
be inferrable, deterministic, and agree with `evaluate_batch`.

`evaluate_batch` must give exactly the results of `evaluate` unless `rtol` is
positive. A problem that evaluates batches on a GPU, in lower precision, agrees
only up to rounding: with `rtol` the fitness and descriptor values then have to
match within the relative tolerance `rtol`, and within `rtol` in absolute terms
for values near zero.
"""
function test_problem(prob::QD.AbstractQDProblem; rng::AbstractRNG=Xoshiro(1), n::Integer=5,
                      rtol::Real=0)
    @testset "test_problem($(nameof(typeof(prob))))" begin
        bounds = QD.descriptor_bounds(prob)
        @testset "descriptor_bounds" begin
            @test bounds isa Tuple
            @test !isempty(bounds)
            @test all(b -> length(b) == 2 && b[1] < b[2], bounds)
        end

        xs = [QD.random_solution(rng, prob) for _ in 1:n]
        @testset "random_solution" begin
            @test isconcretetype(eltype(xs))
            @test QD.random_solution(Xoshiro(7), prob) == QD.random_solution(Xoshiro(7), prob)
        end

        @testset "evaluate" begin
            for x in xs
                ev = @inferred QD.evaluate(prob, x)
                @test ev isa QD.Evaluation
                @test ev.fitness isa Union{Real,AbstractVector{<:Real}}
                @test ev.descriptor isa AbstractVector{<:Real}
                @test length(ev.descriptor) == length(bounds)
                @test all(i -> bounds[i][1] <= ev.descriptor[i] <= bounds[i][2],
                          eachindex(bounds))
                @test QD.evaluate(prob, x) == ev
            end
        end

        @testset "evaluate_batch" begin
            evs = QD.evaluate_batch(prob, xs)
            @test evs isa AbstractVector{<:QD.Evaluation}
            @test length(evs) == n
            reference = [QD.evaluate(prob, x) for x in xs]
            if rtol == 0
                @test evs == reference
            else
                @test all(zip(evs, reference)) do (a, b)
                    isapprox(a.fitness, b.fitness; rtol, atol=rtol) &&
                        isapprox(a.descriptor, b.descriptor; rtol, atol=rtol)
                end
            end
        end

        @testset "fitness_offset" begin
            @test QD.fitness_offset(prob) isa Real
        end
    end
    return nothing
end

"""
    test_archive(make_archive, prob; rng=Xoshiro(1), n=200, check_allocations=true)

Check that the archives built by `make_archive()` meet the contract of
[`AbstractArchive`](@ref). `make_archive` is called without arguments and must
return a new, empty archive each time. `prob` supplies solutions and
evaluations; its descriptors must lie inside the archive's bounds.

The checks cover an empty archive, the status and `improvement` of each kind
of insertion, `length`, `capacity`, `cell_index`, `sample_elites`, `elites`,
`add_batch!`, and `empty!` when the archive defines it. With
`check_allocations`, `cell_index` must not allocate, and neither must `add!`
when the genotype type is `isbits`.
"""
function test_archive(make_archive, prob::QD.AbstractQDProblem;
                      rng::AbstractRNG=Xoshiro(1), n::Integer=200,
                      check_allocations::Bool=true)
    archive = make_archive()
    A = typeof(archive)
    G, F, D = _archive_types(archive)
    @testset "test_archive($(nameof(A)))" begin
        @testset "types" begin
            @test archive isa QD.AbstractArchive{G,F,D}
            @test isconcretetype(A)
            @test QD.capacity(archive) isa Integer
            @test QD.capacity(archive) >= 1
        end

        @testset "empty archive" begin
            @test length(archive) == 0
            @test isempty(collect(QD.elites(archive)))
        end

        xs = [QD.random_solution(rng, prob) for _ in 1:n]
        evs = QD.evaluate_batch(prob, xs)

        @testset "cell_index" begin
            for ev in evs
                i = @inferred QD.cell_index(archive, ev.descriptor)
                @test i isa Integer
                @test 1 <= i <= QD.capacity(archive)
                @test QD.cell_index(archive, ev.descriptor) == i
            end
            if check_allocations
                @test allocations(QD.cell_index, archive, evs[1].descriptor) == 0
            end
        end

        @testset "add!" begin
            a = make_archive()
            ev = evs[1]
            r = @inferred QD.add!(a, xs[1], ev)
            @test r isa QD.AddResult{F}
            @test r.status == QD.NewCell
            @test length(a) == 1

            r = QD.add!(a, xs[1], ev)
            @test r.status == QD.Rejected
            @test r.improvement <= 0
            @test length(a) == 1

            worse = QD.Evaluation(_shift_fitness(ev.fitness, -1), ev.descriptor)
            r = QD.add!(a, xs[1], worse)
            @test r.status == QD.Rejected
            @test r.improvement < 0

            better = QD.Evaluation(_shift_fitness(ev.fitness, 1), ev.descriptor)
            r = QD.add!(a, xs[2], better)
            @test r.status == QD.Improved
            @test r.improvement > 0
            @test length(a) == 1
            @test only(collect(QD.elites(a)))[1] == convert(G, xs[2])

            if check_allocations && isbitstype(G)
                @test allocations(QD.add!, a, xs[1], ev) == 0
            end
        end

        @testset "add_batch!" begin
            a = make_archive()
            b = make_archive()
            rs = QD.add_batch!(a, xs, evs)
            @test rs isa AbstractVector{<:QD.AddResult}
            @test length(rs) == n
            @test rs == [QD.add!(b, x, ev) for (x, ev) in zip(xs, evs)]
            @test length(a) == length(b)
            @test 1 <= length(a) <= QD.capacity(a)
            @test count(r -> r.status == QD.NewCell, rs) == length(a)
        end

        @testset "elites and sample_elites" begin
            a = make_archive()
            QD.add_batch!(a, xs, evs)
            stored = collect(QD.elites(a))
            @test length(stored) == length(a)
            @test all(p -> p[1] isa G && p[2] isa QD.Evaluation, stored)
            @test all(p -> QD.cell_index(a, p[2].descriptor) in 1:QD.capacity(a), stored)
            @test length(unique(QD.cell_index(a, p[2].descriptor) for p in stored)) == length(stored)

            parents = @inferred QD.sample_elites(Xoshiro(3), a, 25)
            @test parents isa Vector{G}
            @test length(parents) == 25
            @test all(p -> any(q -> q[1] == p, stored), parents)
            @test QD.sample_elites(Xoshiro(3), a, 25) == parents
        end

        @testset "empty!" begin
            if hasmethod(empty!, Tuple{A})
                a = make_archive()
                QD.add_batch!(a, xs, evs)
                empty!(a)
                @test length(a) == 0
                @test isempty(collect(QD.elites(a)))
                @test QD.add!(a, xs[1], evs[1]).status == QD.NewCell
            end
        end
    end
    return nothing
end

_archive_types(::QD.AbstractArchive{G,F,D}) where {G,F,D} = (G, F, D)

_shift_fitness(f::Real, by) = f + by
_shift_fitness(f::AbstractVector, by) = f .+ by

"""
    test_variation(op, parent; parent2=nothing, rng=Xoshiro(1))

Check that the variation operator `op` meets the contract of
[`AbstractVariation`](@ref) for the genotype `parent`. Without `parent2` the
operator is a mutation, `vary(rng, op, parent)`, and `n_parents(op)` must be
`1`; with it, a crossover, `vary(rng, op, parent, parent2)`, and `n_parents(op)`
must be `2`.

The call must be inferrable, leave its parents unchanged, and give the same
child for the same RNG state. A new child must not be the parent object.
"""
function test_variation(op::QD.AbstractVariation, parent;
                        parent2=nothing, rng::AbstractRNG=Xoshiro(1))
    @testset "test_variation($(nameof(typeof(op))))" begin
        parents = parent2 === nothing ? (parent,) : (parent, parent2)
        @test QD.n_parents(op) == length(parents)
        saved = map(deepcopy, parents)
        child = @inferred QD.vary(Xoshiro(5), op, parents...)
        @test parents == saved
        @test QD.vary(Xoshiro(5), op, parents...) == child
        @test !any(p -> child === p, parents)
        @test typeof(QD.vary(Xoshiro(6), op, parents...)) == typeof(child)
    end
    return nothing
end

"""
    test_emitter(emitter, archive, prob; rng=Xoshiro(1))

Check that `emitter` meets the contract of [`AbstractEmitter`](@ref). `archive`
must be an archive for `prob` that already holds at least one solution. The
check modifies both `emitter` and `archive`.

`batch_size` must be a positive integer, and `ask` must return that many
genotypes of the archive's genotype type, inferrably. The batch must then pass
through `evaluate_batch`, `add_batch!` and `tell!`.
"""
function test_emitter(emitter::QD.AbstractEmitter, archive::QD.AbstractArchive{G},
                      prob::QD.AbstractQDProblem; rng::AbstractRNG=Xoshiro(1)) where {G}
    @testset "test_emitter($(nameof(typeof(emitter))))" begin
        @test length(archive) >= 1
        @test QD.batch_size(emitter) isa Integer
        @test QD.batch_size(emitter) >= 1

        xs = @inferred QD.ask(Xoshiro(5), emitter, archive)
        @test xs isa Vector{G}
        @test length(xs) == QD.batch_size(emitter)

        @testset "tell!" begin
            evs = QD.evaluate_batch(prob, xs)
            results = QD.add_batch!(archive, xs, evs)
            @test QD.tell!(emitter, xs, evs, results) === nothing
        end
    end
    return nothing
end

end
