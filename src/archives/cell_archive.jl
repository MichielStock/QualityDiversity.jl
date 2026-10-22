"""
    OutsidePolicy

Supertype of the policies that tell a [`CellArchive`](@ref) what to do with a
descriptor outside its bounds: `ThrowOutside()`, `ClampOutside()` and `RejectOutside()`.
"""
abstract type OutsidePolicy end

"Throw an `ArgumentError` for a descriptor outside the bounds."
struct ThrowOutside <: OutsidePolicy end

"Use the nearest cell for a descriptor outside the bounds."
struct ClampOutside <: OutsidePolicy end

"Discard a solution whose descriptor is outside the bounds."
struct RejectOutside <: OutsidePolicy end

"""
    CellArchive{G,F,D,P}

Supertype of the archives that divide descriptor space into a fixed number of
cells and keep the highest-fitness solution found in each: [`GridArchive`](@ref),
[`SlidingBoundariesArchive`](@ref), [`CVTArchive`](@ref) and [`CategoricalArchive`](@ref). They differ in how a
descriptor is assigned to a cell and share everything else: storage, `add!`,
`add_batch!`, `sample_elites`, `elites`, `length`, `capacity`, `empty!` and the
metrics, including the running totals that make `qd_score`, `max_fitness` and
`mean_fitness` constant-time.

A subtype defines [`cell_index`](@ref QualityDiversity.cell_index), returning `0`
for a descriptor its `outside` policy rejects, and has these fields, which the
shared methods use:

- `outside`: an [`OutsidePolicy`](@ref)
- `occupied::BitVector`, `fitness::Vector{F}`, `descriptors::Vector{D}`,
  `genotypes::Vector{G}`, `phenotypes::Vector{P}`, `occupied_idx::Vector{Int}`:
  one entry per cell
- `fitness_sum::Float64`, `fitness_sum_error::Float64`, `fitness_max::F`: the
  running totals described under Metrics in [`GridArchive`](@ref)
"""
abstract type CellArchive{G,F,D,P} <: AbstractArchive{G,F,D,P} end

Base.length(archive::CellArchive) = length(archive.occupied_idx)

capacity(archive::CellArchive) = length(archive.occupied)

Base.empty!(archive::CellArchive) = clear_cells!(archive)

# Empties every cell and resets the running totals.
function clear_cells!(archive::CellArchive{G,F}) where {G,F}
    fill!(archive.occupied, false)
    empty!(archive.occupied_idx)
    archive.fitness_sum = 0.0
    archive.fitness_sum_error = 0.0
    archive.fitness_max = zero(F)
    return archive
end

# Returns the component to use, or NaN to reject the whole descriptor.
function resolve_outside(::ThrowOutside, i, x, lo, hi)
    throw(ArgumentError("descriptor component $i is $x, outside the archive bounds [$lo, $hi]"))
end

function resolve_outside(::ClampOutside, i, x, lo, hi)
    isnan(x) && throw(ArgumentError("descriptor component $i is NaN and cannot be clamped"))
    return clamp(x, lo, hi)
end

resolve_outside(::RejectOutside, i, x, lo, hi) = NaN

# Checks the bounds of descriptor dimension `i`. The width must be finite too: cells are
# found by dividing by it.
function check_bound(i, lo, hi)
    (isfinite(lo) && isfinite(hi) && lo < hi) ||
        throw(ArgumentError("bounds[$i] is ($lo, $hi) but lo must be finite and less than hi"))
    isfinite(hi - lo) ||
        throw(ArgumentError("bounds[$i] is ($lo, $hi), whose width $(hi - lo) is not finite"))
    return nothing
end

"""
    add!(archive::CellArchive, x, ev::Evaluation) -> AddResult

Insert `x` into the cell of `ev.descriptor`. A descriptor outside the bounds is
handled by the archive's `outside` policy; `RejectOutside()` gives a `Rejected` result
with the lowest `improvement`, `-Inf` for a floating-point fitness type. An empty cell takes any solution
(`NewCell`, with `improvement` equal to the fitness); an occupied cell takes it
only if the fitness is strictly higher (`Improved`, otherwise `Rejected`).
For an occupied cell, `improvement` is the fitness minus the occupant's.
"""
add!(archive::CellArchive, x, ev::Evaluation{<:Real}) = add_to_cells!(archive, x, ev)

function add_to_cells!(archive::CellArchive{G,F,D,P}, x, ev::Evaluation{<:Real}) where {G,F,D,P}
    i = cell_index(archive, ev.descriptor)
    fitness = stored_fitness(F, ev.fitness)
    phenotype = stored_phenotype(P, ev.phenotype)
    i == 0 && return outside_result(F)
    result = classify(archive, i, fitness)
    result.status == Rejected || place!(archive, i, x, ev.descriptor, fitness, phenotype)
    return result
end

# Status and improvement of a solution with `fitness` for cell `i`, without
# changing the archive.
function classify(archive::CellArchive, i, fitness)
    archive.occupied[i] || return AddResult(NewCell, fitness)
    previous = archive.fitness[i]
    return AddResult(fitness > previous ? Improved : Rejected, fitness_gain(fitness, previous))
end

# Makes `x` the occupant of cell `i`, keeping the running totals up to date. It
# does not check `fitness` against the occupant.
function place!(archive::CellArchive, i, x, descriptor, fitness, phenotype)
    if !archive.occupied[i]
        archive.occupied[i] = true
        push!(archive.occupied_idx, i)
        _store!(archive, i, x, fitness, descriptor, phenotype)
        _add_to_sum!(archive, Float64(fitness))
        archive.fitness_max = length(archive.occupied_idx) == 1 ? fitness : max(archive.fitness_max, fitness)
        return nothing
    end
    previous = archive.fitness[i]
    _store!(archive, i, x, fitness, descriptor, phenotype)
    # Added and subtracted as two compensated steps: the difference of the two
    # would round for a Float64 fitness, outside the compensation.
    _add_to_sum!(archive, Float64(fitness))
    _add_to_sum!(archive, -Float64(previous))
    archive.fitness_max = max(archive.fitness_max, fitness)
    return nothing
end

"""
    add_batch!(archive::CellArchive, xs, evs) -> Vector{AddResult}

Insert every solution in `xs` with its evaluation from `evs`. The status and
`improvement` of each solution are computed against the archive as it was before
the batch, so they do not depend on the other solutions in the batch: two
solutions that both beat the occupant of a cell are both `Improved`, and two
that fill the same empty cell are both `NewCell`. Of the solutions that enter a
cell, the one with the highest fitness (the first, on a tie) is kept, so the
archive afterwards is the one that adding the solutions one at a time gives.
"""
add_batch!(archive::CellArchive, xs::AbstractVector, evs::AbstractVector) = add_batch_to_cells!(archive, xs, evs)

function add_batch_to_cells!(archive::CellArchive{G,F,D,P}, xs::AbstractVector, evs::AbstractVector) where {G,F,D,P}
    length(xs) == length(evs) ||
        throw(ArgumentError("got $(length(xs)) solutions but $(length(evs)) evaluations"))
    Base.require_one_based_indexing(xs, evs)
    n = length(xs)
    results = Vector{AddResult{F}}(undef, n)
    cells = Vector{Int}(undef, n)
    for k in 1:n
        i = cell_index(archive, evs[k].descriptor)
        fitness = stored_fitness(F, evs[k].fitness)    # throws before the archive changes
        stored_phenotype(P, evs[k].phenotype)
        cells[k] = i
        results[k] = i == 0 ? outside_result(F) : classify(archive, i, fitness)
    end
    # Every result is fixed; placing a solution only if it beats what is in its
    # cell by now leaves the best of the solutions that entered.
    for k in 1:n
        results[k].status == Rejected && continue
        i = cells[k]
        fitness = stored_fitness(F, evs[k].fitness)
        if !archive.occupied[i] || fitness > archive.fitness[i]
            place!(archive, i, xs[k], evs[k].descriptor, fitness, stored_phenotype(P, evs[k].phenotype))
        end
    end
    return results
end

# Neumaier's compensated summation, for any archive with the fields
# `fitness_sum` and `fitness_sum_error`: `error` accumulates the low-order part that
# is lost when `x` is added to `sum`.
function _add_to_sum!(archive::AbstractArchive, x::Float64)
    total = archive.fitness_sum
    updated = total + x
    archive.fitness_sum_error += abs(total) >= abs(x) ? (total - updated) + x : (x - updated) + total
    archive.fitness_sum = updated
    return nothing
end

_fitness_sum(archive::AbstractArchive) = archive.fitness_sum + archive.fitness_sum_error

qd_score(archive::CellArchive; offset=0) = _fitness_sum(archive) + length(archive) * offset

function max_fitness(archive::CellArchive)
    isempty_archive(archive, "max_fitness")
    return archive.fitness_max
end

function mean_fitness(archive::CellArchive)
    isempty_archive(archive, "mean_fitness")
    return _fitness_sum(archive) / length(archive)
end

function _store!(archive::CellArchive, i, x, fitness, descriptor, phenotype)
    archive.genotypes[i] = x
    archive.fitness[i] = fitness
    archive.descriptors[i] = stored_descriptor(archive, descriptor)
    store_phenotype!(archive.phenotypes, i, phenotype)
    return nothing
end

# The descriptor as the archive keeps it; a subtype whose descriptor type differs
# from what its callers pass converts here.
stored_descriptor(::CellArchive, descriptor) = descriptor

"""
    sample_elites(rng, archive::CellArchive, n) -> Vector{G}

`n` genotypes drawn with replacement, uniformly over the occupied cells.
Throws an `ArgumentError` if the archive is empty.
"""
function sample_elites(rng::AbstractRNG, archive::CellArchive{G}, n::Integer) where {G}
    isempty(archive.occupied_idx) &&
        throw(ArgumentError("cannot sample elites from an empty archive"))
    parents = Vector{G}(undef, n)
    for k in 1:n
        parents[k] = archive.genotypes[rand(rng, archive.occupied_idx)]
    end
    return parents
end

function elites(archive::CellArchive)
    return ((archive.genotypes[i], Evaluation(archive.fitness[i], archive.descriptors[i], archive.phenotypes[i]))
            for i in archive.occupied_idx)
end
