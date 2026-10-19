# ── Problem ───────────────────────────────────────────────

"""
    AbstractQDProblem

Supertype of quality-diversity problems. A problem says how to score a
solution and what a random solution looks like.

Fitness is maximized. A problem that minimizes returns the negated value.

# Required methods

- [`evaluate(prob, x)`](@ref QualityDiversity.evaluate) → [`Evaluation`](@ref)
- [`random_solution(rng, prob)`](@ref QualityDiversity.random_solution)
- [`descriptor_bounds(prob)`](@ref QualityDiversity.descriptor_bounds), or
  [`descriptor_categories(prob)`](@ref QualityDiversity.descriptor_categories)
  for a problem whose descriptors take discrete values

# Optional methods

- [`evaluate_batch(prob, xs)`](@ref QualityDiversity.evaluate_batch)
- [`fitness_offset(prob)`](@ref QualityDiversity.fitness_offset)
"""
abstract type AbstractQDProblem end

"""
    AbstractStochasticQDProblem

Supertype of problems whose evaluation is random: scoring the same solution twice can
give different fitness or descriptors. It is a subtype of [`AbstractQDProblem`](@ref).
Instead of `evaluate(prob, x)`, such a problem defines

- [`evaluate(rng, prob, x)`](@ref QualityDiversity.evaluate) → [`Evaluation`](@ref),
  which draws all its randomness from `rng`,

next to the other required methods of [`AbstractQDProblem`](@ref).

A run stays reproducible from its seed, whatever the number of threads. Before a batch
is evaluated, [`evaluate_batch(rng, prob, xs)`](@ref QualityDiversity.evaluate_batch)
draws one seed per solution from the generator of the run, in the order of the batch,
and gives every solution its own `Xoshiro` generator, so which thread evaluates a
solution makes no difference. Deterministic problems do not use the generator of the run.

[`StochasticQDProblem`](@ref) wraps a function, and [`Sampled`](@ref) evaluates a
stochastic problem several times and returns the average.
"""
abstract type AbstractStochasticQDProblem <: AbstractQDProblem end

"""
    evaluate(prob::AbstractQDProblem, x) -> Evaluation
    evaluate(rng::AbstractRNG, prob::AbstractStochasticQDProblem, x) -> Evaluation

Fitness and descriptor of one solution `x`, and its phenotype if the problem computes
one; see [`Evaluation`](@ref). Required. A stochastic problem takes the
generator that it draws its randomness from as the first argument; see
[`AbstractStochasticQDProblem`](@ref).
"""
function evaluate end

function evaluate(prob::AbstractStochasticQDProblem, x)
    throw(ArgumentError("$(nameof(typeof(prob))) is a stochastic problem: call evaluate(rng, prob, x) with a random number generator"))
end

"""
    evaluate_batch(prob::AbstractQDProblem, xs) -> Vector{<:Evaluation}
    evaluate_batch(rng::AbstractRNG, prob::AbstractQDProblem, xs) -> Vector{<:Evaluation}

Evaluate every solution in `xs`, returning one [`Evaluation`](@ref) per
solution in the same order. Optional: a problem specializes it to evaluate a
batch at once, for example on a GPU or through a call into Python.

The default calls [`evaluate`](@ref QualityDiversity.evaluate) on each solution
in parallel with `Threads.@threads`, so `evaluate` must be thread-safe.

To evaluate on a GPU, a problem type defines this method: it copies the batch
to the device, runs array operations or a kernel, and copies the fitness and
descriptor values back. `examples/gpu_evaluation.jl` does this for any array
type of the GPU packages (CUDA.jl, Metal.jl, AMDGPU.jl, oneAPI.jl). The
package itself does not depend on any of them.

The candidates of a [`DeviceCellArchive`](@ref) are a [`DeviceBatch`](@ref), whose
solutions are the columns of a device matrix. For them the default runs
`evaluate(prob, x)` in a kernel, one solution per thread (for a [`QDProblem`](@ref),
its function), and returns an [`EvaluationBatch`](@ref) on the device. That code must
then compile for the device: no allocation, no error message built from values, and no
`Float64` on a device that lacks it. A problem can instead define
`evaluate_batch(prob::MyProblem, xs::DeviceBatch)` and work on `parent(xs)`.

The method with a generator is the one that [`step!`](@ref) calls, with the generator
of the run. For a deterministic problem it calls `evaluate_batch(prob, xs)` and leaves
the generator untouched. For an [`AbstractStochasticQDProblem`](@ref) it draws one seed
per solution from `rng`, in the order of the batch, and evaluates solution `i` with
`evaluate(Xoshiro(seed_i), prob, xs[i])`, so the results do not depend on the number
of threads. A stochastic problem can define its own method with a generator, for
example to evaluate a batch on a device.
"""
function evaluate_batch end

evaluate_batch(prob::AbstractQDProblem, xs::AbstractVector) = default_evaluate_batch(prob, xs)

# The default of `evaluate_batch`, chosen by the type of the batch: a problem that
# defines `evaluate_batch(prob::MyProblem, xs::AbstractVector)` takes every batch.
function default_evaluate_batch(prob::AbstractQDProblem, xs::AbstractVector)
    T = Base.promote_op(evaluate, typeof(prob), eltype(xs))
    evs = Vector{T}(undef, length(xs))
    Threads.@threads for i in eachindex(xs)
        evs[i] = evaluate(prob, xs[i])
    end
    return evs
end

evaluate_batch(rng::AbstractRNG, prob::AbstractQDProblem, xs::AbstractVector) =
    evaluate_batch(prob, xs)

function evaluate_batch(prob::AbstractStochasticQDProblem, xs::AbstractVector)
    throw(ArgumentError("$(nameof(typeof(prob))) is a stochastic problem: call evaluate_batch(rng, prob, xs) with a random number generator"))
end

function evaluate_batch(rng::AbstractRNG, prob::AbstractStochasticQDProblem, xs::AbstractVector)
    # One draw at a time: the seed of a solution is the draw at its position, whatever the
    # length of the batch.
    seeds = [rand(rng, UInt64) for _ in xs]
    T = Base.promote_op(evaluate, Random.Xoshiro, typeof(prob), eltype(xs))
    evs = Vector{T}(undef, length(xs))
    Threads.@threads for i in eachindex(xs)
        evs[i] = evaluate(Random.Xoshiro(seeds[i]), prob, xs[i])
    end
    return evs
end

"""
    evaluations_per_solution(prob::AbstractQDProblem) -> Int

How many times the underlying function is evaluated to score one solution. Optional;
the default is `1`. [`Sampled`](@ref) returns its number of samples, and the budget of a
run counts these evaluations.
"""
function evaluations_per_solution end

evaluations_per_solution(::AbstractQDProblem) = 1

# The evaluation of `x` inside a device kernel, for the default `evaluate_batch` of a
# `DeviceBatch`. A problem whose `evaluate` checks its input with a message built
# from values defines it without the check.
device_evaluate(prob::AbstractQDProblem, x) = evaluate(prob, x)

"""
    random_solution(rng::AbstractRNG, prob::AbstractQDProblem)

One random genotype, used to initialize the archive. Required.
"""
function random_solution end

"""
    descriptor_bounds(prob::AbstractQDProblem)

Per-dimension `(lo, hi)` bounds of the descriptor space, as a tuple of
tuples. Archives use them to lay out their cells. Required, unless the problem
defines [`descriptor_categories`](@ref QualityDiversity.descriptor_categories).

    descriptor_bounds(archive::AbstractArchive)

The same for an archive that covers a box of descriptor space, as [`GridArchive`](@ref),
[`CVTArchive`](@ref) and [`ThresholdArchive`](@ref) do. Optional; the
[`DirectionRanker`](@ref)s need it.
"""
function descriptor_bounds end

"""
    descriptor_categories(prob::AbstractQDProblem)

Per-dimension collections of the values a discrete descriptor can take, as a
tuple, for example `(0:5, [:alkane, :aromatic, :ring])`. A problem that defines
it returns descriptors as tuples with one entry from each collection, such as
`(3, :ring)`, and is used with a [`CategoricalArchive`](@ref). Optional: a
problem with real-valued descriptors defines
[`descriptor_bounds`](@ref QualityDiversity.descriptor_bounds) instead.
"""
function descriptor_categories end

# Number of descriptor dimensions, from the categories if the problem has them
# and from the bounds otherwise.
function descriptor_ndims(prob::AbstractQDProblem)
    hasmethod(descriptor_categories, Tuple{typeof(prob)}) && return length(descriptor_categories(prob))
    return length(descriptor_bounds(prob))
end

"""
    reference_point(prob::AbstractQDProblem)

A point that lies below the fitness values of interest of a problem with several
objectives, as the corner from which a [`ParetoGridArchive`](@ref) measures
hypervolume. Optional: [`MOME`](@ref) takes it from the problem unless it is given.

    reference_point(archive::ParetoGridArchive)

The reference point of the archive.
"""
function reference_point end

function reference_point(prob::AbstractQDProblem)
    throw(ArgumentError("$(nameof(typeof(prob))) does not define reference_point: define it for the problem, or pass the reference point as a keyword"))
end

"""
    fitness_offset(prob::AbstractQDProblem)

Offset that makes every attainable fitness non-negative, as the default
`offset` of [`qd_score`](@ref). Optional; the default is `0`.
"""
function fitness_offset end

fitness_offset(::AbstractQDProblem) = 0

# ── Archive ───────────────────────────────────────────────

"""
    AbstractArchive{G,F,D,P}

Supertype of archives, which store the best solution found for each region of
descriptor space. The type parameters are the genotype `G`, the fitness `F`,
the descriptor `D` and the phenotype `P`, so that code downstream of an archive
is concretely typed. `P` is `Nothing` for an archive that stores no phenotype;
see [`Evaluation`](@ref). An archive with a phenotype type other than `Nothing`
stores the phenotype of every solution it takes and throws for an evaluation
without one.

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
abstract type AbstractArchive{G,F,D,P} end

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
order. An archive may instead compute every result against its state before the
batch, so that a result does not depend on the other solutions in the batch, as
the cell archives and [`ThresholdArchive`](@ref) do; it documents this.
"""
function add_batch! end

function add_batch!(archive::AbstractArchive{G,F,D,P}, xs::AbstractVector, evs::AbstractVector) where {G,F,D,P}
    length(xs) == length(evs) ||
        throw(ArgumentError("got $(length(xs)) solutions but $(length(evs)) evaluations"))
    foreach(ev -> stored_phenotype(P, ev.phenotype), evs)    # throws before the archive changes
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

An archive returns `0`, which is not a cell, for a descriptor it has no cell
for: one that a `RejectOutside()` policy discards, or any descriptor while an
[`UnstructuredArchive`](@ref) is empty.
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
The evaluation holds the stored phenotype, `nothing` for an archive without one.
"""
function elites end

"""
    capacity(archive::AbstractArchive)

Number of cells in `archive`, occupied or not. Required.
"""
function capacity end

"""
    result_archive(archive::AbstractArchive)

The archive that holds the best solutions found, which [`solve`](@ref) returns and
whose metrics a run records. Optional: the default is `archive` itself. An archive
whose elites can get worse, such as a [`ThresholdArchive`](@ref), returns a
separate archive that keeps the best solution of each cell.
"""
function result_archive end

result_archive(archive::AbstractArchive) = archive

# A summary instead of the stored arrays, which can hold millions of numbers.
function Base.show(io::IO, ::MIME"text/plain", archive::AbstractArchive)
    print(io, nameof(typeof(archive)), " with ", length(archive), " of ", capacity(archive), " cells occupied")
end

# ── Variation ─────────────────────────────────────────────

"""
    AbstractVariation

Supertype of variation operators, which turn one or two parent genotypes into
a child.

# Required methods

A mutation defines
[`vary(rng, op, parent)`](@ref QualityDiversity.vary); a crossover defines
[`vary(rng, op, p1, p2)`](@ref QualityDiversity.vary) and
[`n_parents(op)`](@ref QualityDiversity.n_parents) returning `2`.

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

"""
    n_parents(op::AbstractVariation) -> Int

Number of parents `op` combines: `1` for a mutation, which is the default, and
`2` for a crossover. [`VariationEmitter`](@ref) draws that many parents from the
archive for each child. The value must be known from the type of `op`, so that
the choice is made at compile time.
"""
function n_parents end

n_parents(::AbstractVariation) = 1

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
- [`restarts(emitter)`](@ref QualityDiversity.restarts)
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
an emitter without state needs no method. An emitter that keeps state about the
batch it asked for may require that `tell!` follows that `ask`.
"""
function tell! end

tell!(::AbstractEmitter, xs, evs, results) = nothing

"""
    ask_gradients(rng::AbstractRNG, emitter::AbstractEmitter, archive::AbstractArchive) -> Union{Nothing,Vector{G}}

Solutions whose gradients the emitter needs before it can [`ask`](@ref QualityDiversity.ask),
or `nothing`. Optional: the default returns `nothing`, and an emitter that steps along
gradients, such as a [`GradientArborescenceEmitter`](@ref), returns its points. Every
round, [`step!`](@ref) evaluates them with
[`evaluate_gradient_batch`](@ref QualityDiversity.evaluate_gradient_batch), adds them to
the archive, counts them against the budget, and calls
[`tell_gradients!`](@ref QualityDiversity.tell_gradients!) before it calls `ask`.
"""
function ask_gradients end

ask_gradients(::AbstractRNG, ::AbstractEmitter, ::AbstractArchive) = nothing

"""
    tell_gradients!(emitter::AbstractEmitter, xs, evs, jacobians, results)

Report the solutions `xs` that [`ask_gradients`](@ref QualityDiversity.ask_gradients)
returned, their evaluations, their Jacobians (see
[`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient)) and the
[`AddResult`](@ref)s from the archive. Required for an emitter whose `ask_gradients`
returns solutions.
"""
function tell_gradients! end

# Whether the emitter takes gradients through `ask_gradients`; a `BanditScheduler` cannot
# hold such an emitter.
uses_gradients(::AbstractEmitter) = false

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

function descriptor_bounds(archive::AbstractArchive)
    throw(ArgumentError("$(nameof(typeof(archive))) does not report the bounds of its descriptor space"))
end
