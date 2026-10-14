"""
    ThresholdArchive(archive::CellArchive; learning_rate=1.0, threshold_min=-Inf)

Archive for CMA-MAE: a cell archive in which each cell has a fitness threshold
that rises slowly as solutions enter the cell, instead of jumping to the best
fitness found. `archive` decides which cell a descriptor belongs to, so any
[`CellArchive`](@ref) works: a [`GridArchive`](@ref), a [`CVTArchive`](@ref) or a
[`CategoricalArchive`](@ref). It must be empty and have a floating-point
fitness type, and the new archive uses it without copying.

A solution with fitness `f` enters its cell if `f` is strictly greater than the
cell's threshold `t`, and then replaces the cell's elite. The threshold becomes
`(1 - learning_rate) * t + learning_rate * f`. The `improvement` of the
[`AddResult`](@ref) is `f - t`, which is what CMA-ME style emitters rank
candidates by; for a rejected solution it is zero or negative.

- Every threshold starts at `threshold_min`. For `-Inf`, the default, an empty
  cell accepts any solution, reports `improvement == f`, and sets its threshold
  to `f`.
- With `learning_rate=1`, the threshold of a cell is the fitness of its elite and
  the archive behaves exactly like the cell archive it wraps.
- With a smaller `learning_rate`, the threshold stays below the elite's fitness,
  so a solution that is better than the threshold but worse than the elite
  replaces the elite. The elite's fitness can therefore decrease, and the
  archive's `max_fitness` with it. A rate of `0` keeps each threshold at its
  initial value.

`learning_rate` must lie in `[0, 1]`. [`thresholds`](@ref QualityDiversity.thresholds)
returns the thresholds, one per cell.

Cells, sampling, `elites`, `length`, `capacity` and `cell_index` are those of the
wrapped archive. `qd_score` and `mean_fitness` take constant time. `max_fitness`
does too, except directly after the best elite was replaced by a worse solution,
when the next call makes a pass over the elites.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = ThresholdArchive(GridArchive((10, 10), ((0, 1), (0, 1))); learning_rate=0.5);

julia> length(archive), QualityDiversity.capacity(archive)
(0, 100)
```
"""
mutable struct ThresholdArchive{G,F<:AbstractFloat,D,A<:CellArchive{G,F,D}} <: AbstractArchive{G,F,D}
    const inner::A
    const thresholds::Vector{F}
    const learning_rate::F
    const threshold_min::F
    # Set when the elite with the highest fitness was replaced by a worse one, so
    # that `inner.fitness_max` is only an upper bound.
    max_stale::Bool

    function ThresholdArchive{G,F,D,A}(inner::A, learning_rate, threshold_min) where {G,F<:AbstractFloat,D,A<:CellArchive{G,F,D}}
        length(inner) == 0 ||
            throw(ArgumentError("the archive to wrap must be empty but holds $(length(inner)) solutions"))
        0 <= learning_rate <= 1 ||
            throw(ArgumentError("learning_rate must be in [0, 1], got $learning_rate"))
        (isnan(threshold_min) || threshold_min == Inf) &&
            throw(ArgumentError("threshold_min must be a number below Inf, got $threshold_min"))
        return new{G,F,D,A}(inner, fill(convert(F, threshold_min), capacity(inner)),
                            convert(F, learning_rate), convert(F, threshold_min), false)
    end
end

function ThresholdArchive(inner::CellArchive{G,F,D}; learning_rate::Real=1.0,
                          threshold_min::Real=-Inf) where {G,F<:AbstractFloat,D}
    return ThresholdArchive{G,F,D,typeof(inner)}(inner, learning_rate, threshold_min)
end

"""
    thresholds(archive::ThresholdArchive) -> Vector

The fitness threshold of each cell, in the order of the cell indices. A cell
that has never accepted a solution holds `threshold_min`.
"""
thresholds(archive::ThresholdArchive) = archive.thresholds

Base.length(archive::ThresholdArchive) = length(archive.inner)

capacity(archive::ThresholdArchive) = capacity(archive.inner)

function Base.empty!(archive::ThresholdArchive)
    empty!(archive.inner)
    fill!(archive.thresholds, archive.threshold_min)
    archive.max_stale = false
    return archive
end

cell_index(archive::ThresholdArchive, d) = cell_index(archive.inner, d)

sample_elites(rng::AbstractRNG, archive::ThresholdArchive, n::Integer) =
    sample_elites(rng, archive.inner, n)

elites(archive::ThresholdArchive) = elites(archive.inner)

"""
    add!(archive::ThresholdArchive, x, ev::Evaluation) -> AddResult

Insert `x` into the cell of `ev.descriptor` if its fitness exceeds the cell's
threshold, and raise the threshold; see [`ThresholdArchive`](@ref). The status
is `NewCell` for an empty cell, `Improved` for an occupied one and `Rejected`
otherwise. A descriptor that the wrapped archive discards gives `Rejected` with
zero `improvement`.
"""
function add!(archive::ThresholdArchive{G,F}, x, ev::Evaluation{<:Real}) where {G,F}
    inner = archive.inner
    i = cell_index(inner, ev.descriptor)
    fitness = convert(F, ev.fitness)
    i == 0 && return AddResult(Rejected, zero(F))
    rate = archive.learning_rate
    if !inner.occupied[i]
        floor = archive.threshold_min
        floor == -Inf && return add_to_empty_cell!(archive, i, x, ev, fitness, fitness, fitness)
        improvement = fitness - floor
        improvement > 0 || return AddResult(Rejected, improvement)
        return add_to_empty_cell!(archive, i, x, ev, fitness, (1 - rate) * floor + rate * fitness, improvement)
    end
    threshold = archive.thresholds[i]
    improvement = fitness - threshold
    improvement > 0 || return AddResult(Rejected, improvement)
    previous = inner.fitness[i]
    archive.thresholds[i] = (1 - rate) * threshold + rate * fitness
    _store!(inner, i, x, fitness, ev.descriptor)
    _add_to_sum!(inner, Float64(fitness) - Float64(previous))
    if fitness >= inner.fitness_max
        inner.fitness_max = fitness
        archive.max_stale = false
    elseif previous == inner.fitness_max
        archive.max_stale = true
    end
    return AddResult(Improved, improvement)
end

function add_to_empty_cell!(archive::ThresholdArchive, i, x, ev, fitness, threshold, improvement)
    inner = archive.inner
    inner.occupied[i] = true
    push!(inner.occupied_idx, i)
    archive.thresholds[i] = threshold
    _store!(inner, i, x, fitness, ev.descriptor)
    _add_to_sum!(inner, Float64(fitness))
    if length(inner.occupied_idx) == 1 || fitness >= inner.fitness_max
        inner.fitness_max = fitness
        archive.max_stale = false
    end
    return AddResult(NewCell, improvement)
end

qd_score(archive::ThresholdArchive; offset=0) = qd_score(archive.inner; offset)

mean_fitness(archive::ThresholdArchive) = mean_fitness(archive.inner)

function max_fitness(archive::ThresholdArchive)
    inner = archive.inner
    isempty_archive(inner, "max_fitness")
    if archive.max_stale
        inner.fitness_max = maximum(inner.fitness[i] for i in inner.occupied_idx)
        archive.max_stale = false
    end
    return inner.fitness_max
end
