# ── Problem ───────────────────────────────────────────────

"""
    AbstractQDProblem

Supertype of quality-diversity problems. A problem says how to score a
solution and what a random solution looks like.

Fitness is maximized. A problem that minimizes returns the negated value.

# Required methods

- [`evaluate(prob, x)`](@ref QualityDiversity.evaluate) → [`Evaluation`](@ref)
- [`random_solution(rng, prob)`](@ref QualityDiversity.random_solution)
- [`descriptor_bounds(prob)`](@ref QualityDiversity.descriptor_bounds)

# Optional methods

- [`evaluate_batch(prob, xs)`](@ref QualityDiversity.evaluate_batch)
- [`fitness_offset(prob)`](@ref QualityDiversity.fitness_offset)
"""
abstract type AbstractQDProblem end

"""
    evaluate(prob::AbstractQDProblem, x) -> Evaluation

Fitness and descriptor of one solution `x`. Required.
"""
function evaluate end

"""
    evaluate_batch(prob::AbstractQDProblem, xs) -> Vector{<:Evaluation}

Evaluate every solution in `xs`, returning one [`Evaluation`](@ref) per
solution in the same order. Optional: a problem specializes it to evaluate a
batch at once, for example on a GPU or through a call into Python.

The default calls [`evaluate`](@ref QualityDiversity.evaluate) on each solution
in parallel with `Threads.@threads`, so `evaluate` must be thread-safe.
"""
function evaluate_batch end

function evaluate_batch(prob::AbstractQDProblem, xs::AbstractVector)
    T = Base.promote_op(evaluate, typeof(prob), eltype(xs))
    evs = Vector{T}(undef, length(xs))
    Threads.@threads for i in eachindex(xs)
        evs[i] = evaluate(prob, xs[i])
    end
    return evs
end

"""
    random_solution(rng::AbstractRNG, prob::AbstractQDProblem)

One random genotype, used to initialize the archive. Required.
"""
function random_solution end

"""
    descriptor_bounds(prob::AbstractQDProblem)

Per-dimension `(lo, hi)` bounds of the descriptor space, as a tuple of
tuples. Archives use them to lay out their cells. Required.
"""
function descriptor_bounds end

"""
    fitness_offset(prob::AbstractQDProblem)

Offset that makes every attainable fitness non-negative, as the default
`offset` of [`qd_score`](@ref). Optional; the default is `0`.
"""
function fitness_offset end

fitness_offset(::AbstractQDProblem) = 0

# ── Archive ───────────────────────────────────────────────

"""
    AbstractArchive{G,F,D}

Supertype of archives, which store the best solution found for each region of
descriptor space. The type parameters are the genotype `G`, the fitness `F`
and the descriptor `D`, so that code downstream of an archive is concretely
typed.

# Required methods

- [`add!(archive, x, ev)`](@ref QualityDiversity.add!) → [`AddResult`](@ref)
- [`cell_index(archive, d)`](@ref QualityDiversity.cell_index)
- [`sample_elites(rng, archive, n)`](@ref QualityDiversity.sample_elites)
- [`elites(archive)`](@ref QualityDiversity.elites)
- [`capacity(archive)`](@ref QualityDiversity.capacity)
- `Base.length(archive)`: the number of occupied cells

# Optional methods

- [`add_batch!(archive, xs, evs)`](@ref QualityDiversity.add_batch!)
- `Base.empty!(archive)`
"""
abstract type AbstractArchive{G,F,D} end

"""
    add!(archive::AbstractArchive, x, ev::Evaluation) -> AddResult

Try to insert the solution `x` with evaluation `ev` into `archive`. Required.

Throws an `ArgumentError` if the dimension of `ev.descriptor` does not match
the archive. Does not allocate for `isbits` genotypes.
"""
function add! end

"""
    add_batch!(archive::AbstractArchive, xs, evs) -> Vector{<:AddResult}

Insert every solution in `xs` with its evaluation from `evs`, returning one
[`AddResult`](@ref) per solution in order. Optional: an archive specializes it
when a batch can be inserted faster than one solution at a time.

The default calls [`add!`](@ref QualityDiversity.add!) on each solution in
order.
"""
function add_batch! end

function add_batch!(archive::AbstractArchive{G,F}, xs::AbstractVector, evs::AbstractVector) where {G,F}
    length(xs) == length(evs) ||
        throw(ArgumentError("got $(length(xs)) solutions but $(length(evs)) evaluations"))
    results = Vector{AddResult{F}}(undef, length(xs))
    for i in eachindex(xs, evs)
        results[i] = add!(archive, xs[i], evs[i])
    end
    return results
end

"""
    cell_index(archive::AbstractArchive, d)

The cell of `archive` that contains descriptor `d`. Required. Does not
allocate.
"""
function cell_index end

"""
    sample_elites(rng::AbstractRNG, archive::AbstractArchive, n) -> Vector{G}

Draw `n` parent genotypes from the occupied cells of `archive`. Required.
Uniform over occupied cells unless the archive documents otherwise.
"""
function sample_elites end

"""
    elites(archive::AbstractArchive)

Iterator over `(genotype, evaluation)` pairs of the occupied cells. Required.
"""
function elites end

"""
    capacity(archive::AbstractArchive)

Number of cells in `archive`, occupied or not. Required.
"""
function capacity end

# ── Variation ─────────────────────────────────────────────

"""
    AbstractVariation

Supertype of variation operators, which turn one or two parent genotypes into
a child.

# Required methods

A mutation defines
[`vary(rng, op, parent)`](@ref QualityDiversity.vary); a crossover defines
[`vary(rng, op, p1, p2)`](@ref QualityDiversity.vary).

A user with a custom genotype defines a new operator type and a `vary` method
for it.
"""
abstract type AbstractVariation end

"""
    vary(rng::AbstractRNG, op::AbstractVariation, parent)
    vary(rng::AbstractRNG, op::AbstractVariation, p1, p2)

One child of `parent`, or of the parents `p1` and `p2`. Does not modify its
parents.
"""
function vary end

# ── Emitter ───────────────────────────────────────────────

"""
    AbstractEmitter

Supertype of emitters, which propose batches of candidate solutions and may
learn from how the archive received them.

# Required methods

- [`ask(rng, emitter, archive)`](@ref QualityDiversity.ask)
- [`batch_size(emitter)`](@ref QualityDiversity.batch_size)

# Optional methods

- [`tell!(emitter, xs, evs, results)`](@ref QualityDiversity.tell!)
"""
abstract type AbstractEmitter end

"""
    ask(rng::AbstractRNG, emitter::AbstractEmitter, archive::AbstractArchive) -> Vector{G}

A batch of `batch_size(emitter)` candidate genotypes. Required.
"""
function ask end

"""
    tell!(emitter::AbstractEmitter, xs, evs, results)

Report the candidates `xs` returned by [`ask`](@ref QualityDiversity.ask),
their evaluations `evs`, and the [`AddResult`](@ref)s from the archive, so the
emitter can update its internal state. Optional: the default does nothing, so
an emitter without state needs no method.
"""
function tell! end

tell!(::AbstractEmitter, xs, evs, results) = nothing

"""
    batch_size(emitter::AbstractEmitter)

Number of candidates `ask` returns per call. Required.
"""
function batch_size end

# ── Algorithm ─────────────────────────────────────────────

"""
    QDAlgorithm(archive, emitters)

A quality-diversity algorithm: an archive and a tuple of emitters. Every
algorithm shares one ask → evaluate → add → tell loop; they differ in which
archive and emitters they combine.

`emitters` is a `Tuple` so that the loop over emitters is unrolled and
type-stable.

The algorithm runs through the CommonSolve.jl interface: `init` creates a
state, `step!` performs one round, `solve!` runs to the budget, and
[`solve`](@ref) does `init` and `solve!` in one call.
"""
struct QDAlgorithm{A<:AbstractArchive,E<:Tuple}
    archive::A
    emitters::E
end
