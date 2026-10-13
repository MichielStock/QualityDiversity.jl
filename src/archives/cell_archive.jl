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
    CellArchive{G,F,D}

Supertype of the archives that divide descriptor space into a fixed number of
cells and keep the highest-fitness solution found in each: [`GridArchive`](@ref)
and [`CVTArchive`](@ref). They differ in how a descriptor is assigned to a cell
and share everything else: storage, `add!`, `add_batch!`, `sample_elites`,
`elites`, `length`, `capacity`, `empty!` and the metrics, including the running
totals that make `qd_score`, `max_fitness` and `mean_fitness` constant-time.

A subtype defines [`cell_index`](@ref QualityDiversity.cell_index), returning `0`
for a descriptor its `outside` policy rejects, and has these fields, which the
shared methods use:

- `outside`: an [`OutsidePolicy`](@ref)
- `occupied::BitVector`, `fitness::Vector{F}`, `descriptors::Vector{D}`,
  `genotypes::Vector{G}`, `occupied_idx::Vector{Int}`: one entry per cell
- `fitness_sum::Float64`, `fitness_sum_error::Float64`, `fitness_max::F`: the
  running totals described under Metrics in [`GridArchive`](@ref)
"""
abstract type CellArchive{G,F,D} <: AbstractArchive{G,F,D} end

Base.length(archive::CellArchive) = length(archive.occupied_idx)

capacity(archive::CellArchive) = length(archive.occupied)

function Base.empty!(archive::CellArchive{G,F}) where {G,F}
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

"""
    add!(archive::CellArchive, x, ev::Evaluation) -> AddResult

Insert `x` into the cell of `ev.descriptor`. A descriptor outside the bounds is
handled by the archive's `outside` policy; `RejectOutside()` gives a `Rejected` result
with zero `improvement`. An empty cell takes any solution
(`NewCell`, with `improvement` equal to the fitness); an occupied cell takes it
only if the fitness is strictly higher (`Improved`, otherwise `Rejected`).
For an occupied cell, `improvement` is the fitness minus the occupant's.
"""
function add!(archive::CellArchive{G,F}, x,
              ev::Evaluation{<:Real,<:AbstractVector{<:Real}}) where {G,F}
    i = cell_index(archive, ev.descriptor)
    fitness = convert(F, ev.fitness)
    i == 0 && return AddResult(Rejected, zero(F))
    if !archive.occupied[i]
        archive.occupied[i] = true
        push!(archive.occupied_idx, i)
        _store!(archive, i, x, fitness, ev.descriptor)
        _add_to_sum!(archive, Float64(fitness))
        archive.fitness_max = length(archive.occupied_idx) == 1 ? fitness : max(archive.fitness_max, fitness)
        return AddResult(NewCell, fitness)
    end
    previous = archive.fitness[i]
    improvement = fitness - previous
    if improvement > 0
        _store!(archive, i, x, fitness, ev.descriptor)
        # The difference is taken in Float64, where it is exact for Float32
        # and smaller fitness types.
        _add_to_sum!(archive, Float64(fitness) - Float64(previous))
        archive.fitness_max = max(archive.fitness_max, fitness)
        return AddResult(Improved, improvement)
    end
    return AddResult(Rejected, improvement)
end

# Neumaier's compensated summation: `error` accumulates the low-order part that
# is lost when `x` is added to `sum`.
function _add_to_sum!(archive::CellArchive, x::Float64)
    total = archive.fitness_sum
    updated = total + x
    archive.fitness_sum_error += abs(total) >= abs(x) ? (total - updated) + x : (x - updated) + total
    archive.fitness_sum = updated
    return nothing
end

_fitness_sum(archive::CellArchive) = archive.fitness_sum + archive.fitness_sum_error

qd_score(archive::CellArchive; offset=0) = _fitness_sum(archive) + length(archive) * offset

function max_fitness(archive::CellArchive)
    isempty_archive(archive, "max_fitness")
    return archive.fitness_max
end

function mean_fitness(archive::CellArchive)
    isempty_archive(archive, "mean_fitness")
    return _fitness_sum(archive) / length(archive)
end

function _store!(archive::CellArchive, i, x, fitness, descriptor)
    archive.genotypes[i] = x
    archive.fitness[i] = fitness
    archive.descriptors[i] = descriptor
    return nothing
end

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
    return ((archive.genotypes[i], Evaluation(archive.fitness[i], archive.descriptors[i]))
            for i in archive.occupied_idx)
end

function check_compatible(prob::AbstractQDProblem, archive::CellArchive{G,F,SVector{N,Float64}}) where {G,F,N}
    n = length(descriptor_bounds(prob))
    n == N || throw(ArgumentError("the problem has $n descriptor dimensions but the archive has $N"))
    return nothing
end
