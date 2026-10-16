```@meta
CurrentModule = QualityDiversity
```

# Extending the package

Every component plays one of five roles, and each role is an abstract type with a
short contract: the functions a new type must define, and those it may define. A type
that meets its contract works with every other component and with the generic loop.
`QualityDiversity.Testing` checks that it does.

Extend the interface functions with their qualified names, as you would `Base.show`:

```julia
QualityDiversity.evaluate(prob::MyProblem, x) = ...
```

## A problem

[`QDProblem`](@ref) covers most problems. A type of your own is useful when the
problem has parameters, or to evaluate a batch at once. It defines three functions:

```@example extending
using QualityDiversity
using Random: AbstractRNG, Xoshiro
using StaticArrays: SVector

"Points in the unit square; fitness is closeness to `target`."
struct Target <: AbstractQDProblem
    target::NTuple{2,Float64}
end

function QualityDiversity.evaluate(prob::Target, x)
    point = clamp.(SVector(x[1], x[2]), 0.0, 1.0)     # the descriptor stays within its bounds
    fitness = -hypot(point[1] - prob.target[1], point[2] - prob.target[2])
    return Evaluation(fitness, point)
end

QualityDiversity.random_solution(rng::AbstractRNG, ::Target) = rand(rng, 2)
QualityDiversity.descriptor_bounds(::Target) = ((0.0, 1.0), (0.0, 1.0))
nothing # hide
```

An [`Evaluation`](@ref) holds the fitness, the descriptor, an `SVector` from
StaticArrays.jl (or a tuple, for categories), and an optional phenotype. Optional functions are
[`evaluate_batch`](@ref QualityDiversity.evaluate_batch), to score a batch at once,
[`fitness_offset`](@ref QualityDiversity.fitness_offset), for the QD score, and
[`reference_point`](@ref QualityDiversity.reference_point), for several objectives.
[`test_problem`](@ref QualityDiversity.Testing.test_problem) checks the contract,
including that `evaluate` is inferrable and the descriptors lie within the bounds:

```@example extending
using QualityDiversity.Testing
test_problem(Target((0.3, 0.7)))
nothing # hide
```

The default `evaluate_batch` calls `evaluate` on several threads, so `evaluate` must
be thread-safe: it must not write to shared state without a lock.

## A genotype and its variation

A genotype can be any Julia type. The archives store it as it is, and only the
variation operator looks inside: define an operator type and a `vary` method that
returns a new child and leaves the parent unchanged. The
[graph tutorial](../tutorials/graphs.md) does this for `Graphs.SimpleGraph`. Here is a
mutation for strings:

```@example extending
struct LetterMutation <: AbstractVariation end

function QualityDiversity.vary(rng::AbstractRNG, ::LetterMutation, parent::String)
    letters = collect(parent)
    letters[rand(rng, eachindex(letters))] = rand(rng, 'a':'z')
    return String(letters)
end

test_variation(LetterMutation(), "quality")
nothing # hide
```

A crossover defines `vary(rng, op, p1, p2)` and
[`n_parents`](@ref QualityDiversity.n_parents)`(op) = 2`. Tell the algorithm the
genotype type, so that the archive is concretely typed:
`MAPElites(prob; dims, genotype=String, variation=LetterMutation())`.

## An emitter

An emitter defines [`ask`](@ref QualityDiversity.ask), which returns a batch of
candidates, and [`batch_size`](@ref QualityDiversity.batch_size). It may define
[`tell!`](@ref QualityDiversity.tell!) to learn from the results; the default does
nothing. This one mutates only the fittest elites:

```@example extending
"Mutate parents drawn from the `top` fittest elites."
struct GreedyEmitter{V<:AbstractVariation} <: AbstractEmitter
    op::V
    top::Int
    batch_size::Int
end

QualityDiversity.batch_size(e::GreedyEmitter) = e.batch_size

function QualityDiversity.ask(rng::AbstractRNG, e::GreedyEmitter, archive::AbstractArchive{G}) where {G}
    ranked = sort!(collect(QualityDiversity.elites(archive)); by=elite -> elite[2].fitness, rev=true)
    parents = first.(ranked[1:min(e.top, length(ranked))])
    return G[QualityDiversity.vary(rng, e.op, rand(rng, parents)) for _ in 1:e.batch_size]
end

prob = Target((0.3, 0.7))
archive = GridArchive(prob; dims=(10, 10))
alg = QDAlgorithm(archive, (VariationEmitter(GaussianMutation(0.05); batch_size=20),
                            GreedyEmitter(GaussianMutation(0.01), 5, 20)))
result = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
max_fitness(result.archive)
```

[`test_emitter`](@ref QualityDiversity.Testing.test_emitter) checks the contract
against an archive that holds a few solutions. The batch must be a `Vector` of the
archive's genotype type, and `ask` must be inferrable, which the `G[...]`
comprehension above ensures.

An emitter with state is a `mutable struct`, or holds mutable fields; `init` copies
the emitters, so a run never changes the algorithm it was given.

## An archive

An archive is the most involved role. It defines
[`add!`](@ref QualityDiversity.add!), which returns an [`AddResult`](@ref) with the
status (`NewCell`, `Improved` or `Rejected`) and the improvement over the previous
occupant, [`cell_index`](@ref QualityDiversity.cell_index),
[`sample_elites`](@ref QualityDiversity.sample_elites),
[`elites`](@ref QualityDiversity.elites), [`capacity`](@ref QualityDiversity.capacity)
and `Base.length`, the number of occupied cells. The type parameters of
[`AbstractArchive`](@ref)`{G,F,D,P}` are the genotype, fitness, descriptor and
phenotype types, which emitters use to build concretely typed batches.

A new kind of cell for real-valued descriptors does not need a new archive: subtypes
of [`CellArchive`](@ref QualityDiversity.CellArchive) share the storage, insertion and
metrics, and differ only in `cell_index`. Read the source of
[`GridArchive`](@ref) for a small example, and check the result with
[`test_archive`](@ref QualityDiversity.Testing.test_archive), which also checks that
`cell_index` and `add!` do not allocate.

## Checking conformance in your tests

The checks of `QualityDiversity.Testing` run inside a `@testset`, so they report to
the test runner of the package that calls them:

```julia
using Test, QualityDiversity, QualityDiversity.Testing

@testset "MyPackage" begin
    test_problem(MyProblem())
    test_variation(MyMutation(), example_genotype)
    test_archive(() -> MyArchive(...), MyProblem())
end
```
