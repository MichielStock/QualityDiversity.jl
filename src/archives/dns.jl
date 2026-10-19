"""
    DNSArchive{G,F}(ndims; capacity, k, phenotype=Nothing)
    DNSArchive{G}(ndims; capacity, k, phenotype=Nothing)
    DNSArchive(ndims; capacity, k, phenotype=Nothing)
    DNSArchive(prob::AbstractQDProblem; capacity, k, genotype=Vector{Float64},
               fitness=Float64, phenotype=Nothing)

Archive of Dominated Novelty Search (Bahlous-Boldi, Faldor et al., 2025, "Dominated
Novelty Search: Rethinking Local Competition in Quality-Diversity"; pyribs `DNSArchive`,
QDax `DominatedNoveltyRepertoire`). It keeps a population of at most `capacity`
solutions, with no cells and no bounds on descriptor space.

`ndims` is the number of descriptor dimensions; the last form takes it from
`length(descriptor_bounds(prob))`. `G` is the genotype type, `Vector{Float64}` by
default, and `F<:AbstractFloat` the fitness type, `Float64` by default. Descriptors are
`SVector{ndims,Float64}`, and distances are Euclidean in descriptor units, so
descriptors on very different scales should be normalized by the problem. `phenotype`
is the type of the phenotypes the archive stores, `Nothing` for none; see
[`Evaluation`](@ref).

[`add_batch!`](@ref QualityDiversity.add_batch!) merges the batch into the population
and gives every solution its dominated novelty: the mean distance to its `k` nearest
other solutions whose fitness is at least as high (all of them if there are fewer), or
`Inf` if there is none. The `capacity` solutions with the highest scores survive; on a
tie, a solution of the batch goes before a stored one, and a later one before an
earlier one. A solution of the batch that survives is `NewCell` and one that does not is
`Rejected`, and its `improvement` is its score. Since the population is chosen as a
whole, the result of a solution depends on the rest of the batch. A neighbour of equal
fitness counts as fitter, as in pyribs and QDax, so solutions on a plateau of fitness
still compete on distance; the paper counts strictly fitter neighbours only. `add!` is
a batch of one solution.

[`cell_index`](@ref QualityDiversity.cell_index) is the index of the stored solution
with the nearest descriptor, or `0` while the archive is empty, and
[`capacity`](@ref QualityDiversity.capacity) the population size, so `coverage` is the
fraction of it in use. Descriptors must be finite; a `NaN` or infinite component throws
an `ArgumentError`. Scoring takes time quadratic in the population plus the batch and
keeps only `k` distances per solution. `max_fitness` and `mean_fitness` can decrease,
since a solution can leave the population.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = DNSArchive(2; capacity=100, k=3);

julia> length(archive), QualityDiversity.capacity(archive)
(0, 100)
```
"""
mutable struct DNSArchive{G,F<:AbstractFloat,N,P} <: AbstractArchive{G,F,SVector{N,Float64},P}
    const k::Int
    # One entry per slot; the first `n` are in use.
    const fitness::Vector{F}
    const descriptors::Vector{SVector{N,Float64}}
    const genotypes::Vector{G}
    const phenotypes::Vector{P}
    n::Int
    fitness_sum::Float64
    fitness_sum_error::Float64
    fitness_max::F
    # Workspace for one selection, over the stored solutions followed by the batch.
    const merged_fitness::Vector{F}
    const merged_descriptors::Vector{SVector{N,Float64}}
    const scores::Vector{Float64}
    const order::Vector{Int}
    const kept::Vector{Bool}
    # The `k` smallest distances found so far, in increasing order.
    const nearest::Vector{Float64}

    function DNSArchive{G,F,N,P}(capacity, k) where {G,F<:AbstractFloat,N,P}
        N >= 1 || throw(ArgumentError("ndims must be at least 1, got $N"))
        capacity >= 1 || throw(ArgumentError("capacity must be at least 1, got $capacity"))
        k >= 1 || throw(ArgumentError("k must be at least 1, got $k"))
        # sized for `add!`, a batch of one, so that it does not allocate
        merged = capacity + 1
        return new{G,F,N,P}(k,
                            Vector{F}(undef, capacity),
                            Vector{SVector{N,Float64}}(undef, capacity),
                            Vector{G}(undef, capacity),
                            Vector{P}(undef, capacity),
                            0, 0.0, 0.0, zero(F),
                            Vector{F}(undef, merged), Vector{SVector{N,Float64}}(undef, merged),
                            Vector{Float64}(undef, merged), Vector{Int}(undef, merged),
                            Vector{Bool}(undef, merged), Vector{Float64}(undef, k))
    end
end

function DNSArchive{G,F}(ndims::Integer; capacity::Integer, k::Integer,
                         phenotype::Type=Nothing) where {G,F<:AbstractFloat}
    return DNSArchive{G,F,Int(ndims),phenotype}(Int(capacity), Int(k))
end

DNSArchive{G}(ndims::Integer; kwargs...) where {G} = DNSArchive{G,Float64}(ndims; kwargs...)

DNSArchive(ndims::Integer; kwargs...) = DNSArchive{Vector{Float64}}(ndims; kwargs...)

function DNSArchive(prob::AbstractQDProblem; capacity::Integer, k::Integer,
                    genotype::Type=Vector{Float64}, fitness::Type{<:AbstractFloat}=Float64,
                    phenotype::Type=Nothing)
    return DNSArchive{genotype,fitness}(length(descriptor_bounds(prob)); capacity, k, phenotype)
end

Base.length(archive::DNSArchive) = archive.n

capacity(archive::DNSArchive) = length(archive.fitness)

function Base.empty!(archive::DNSArchive{G,F}) where {G,F}
    archive.n = 0
    archive.fitness_sum = 0.0
    archive.fitness_sum_error = 0.0
    archive.fitness_max = zero(F)
    return archive
end

# `d` as an `SVector`, after checking its dimension and that it is finite.
function checked_descriptor(::DNSArchive{G,F,N}, d::AbstractVector{<:Real}) where {G,F,N}
    length(d) == N ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $N"))
    point = SVector{N,Float64}(d)
    for i in 1:N
        isfinite(point[i]) ||
            throw(ArgumentError("descriptor component $i is $(point[i]), which is not finite"))
    end
    return point
end

"""
    cell_index(archive::DNSArchive, d)

Index of the stored solution whose descriptor is nearest to `d`; the lowest index if two
are equally near. `0`, which is not a slot, while the archive is empty. Throws an
`ArgumentError` if `d` has the wrong number of dimensions or a component that is not
finite.
"""
function cell_index(archive::DNSArchive, d::AbstractVector{<:Real})
    point = checked_descriptor(archive, d)
    best = 0
    best_distance = Inf
    for i in 1:archive.n
        distance = sum(abs2, point - archive.descriptors[i])
        if distance < best_distance
            best, best_distance = i, distance
        end
    end
    return best
end

"""
    add!(archive::DNSArchive, x, ev::Evaluation) -> AddResult

Select the population again from the stored solutions and `x`; see
[`DNSArchive`](@ref).
"""
function add!(archive::DNSArchive, x, ev::Evaluation{<:Real,<:AbstractVector{<:Real}})
    stored = select_population!(archive, (x,), (ev,))
    return dns_result(archive, stored + 1)
end

function add_batch!(archive::DNSArchive{G,F}, xs::AbstractVector, evs::AbstractVector) where {G,F}
    length(xs) == length(evs) ||
        throw(ArgumentError("got $(length(xs)) solutions but $(length(evs)) evaluations"))
    Base.require_one_based_indexing(xs, evs)
    stored = select_population!(archive, xs, evs)
    return [dns_result(archive, stored + j) for j in eachindex(xs)]
end

# The result of merged solution `m` of the last selection.
function dns_result(archive::DNSArchive{G,F}, m) where {G,F}
    return AddResult(archive.kept[m] ? NewCell : Rejected, convert(F, archive.scores[m]))
end

# Merges the batch into the population, keeps the `capacity` solutions with the highest
# dominated novelty and returns how many solutions were stored before. The batch is
# checked before the archive changes.
function select_population!(archive::DNSArchive{G,F,N,P}, xs, evs) where {G,F,N,P}
    stored = archive.n
    merged = stored + length(xs)
    if length(archive.scores) < merged
        resize!(archive.merged_fitness, merged)
        resize!(archive.merged_descriptors, merged)
        resize!(archive.scores, merged)
        resize!(archive.order, merged)
        resize!(archive.kept, merged)
    end
    for i in 1:stored
        archive.merged_fitness[i] = archive.fitness[i]
        archive.merged_descriptors[i] = archive.descriptors[i]
    end
    for j in 1:length(xs)
        ev = evs[j]
        archive.merged_descriptors[stored+j] = checked_descriptor(archive, ev.descriptor)
        archive.merged_fitness[stored+j] = stored_fitness(F, ev.fitness)
        stored_phenotype(P, ev.phenotype)    # throws before the archive changes
    end
    for m in 1:merged
        archive.scores[m] = dominated_novelty(archive, m, merged)
    end

    survivors = min(merged, capacity(archive))
    order = view(archive.order, 1:merged)
    for m in 1:merged
        order[m] = m
    end
    if merged > survivors
        scores = archive.scores
        # highest score first; on a tie the later solution first
        sort!(order; alg=QuickSort, lt=(a, b) -> scores[a] > scores[b] || (scores[a] == scores[b] && a > b))
        sort!(view(order, 1:survivors); alg=QuickSort)
    end
    fill!(view(archive.kept, 1:merged), false)
    # Survivors in increasing merged order fill the slots from the first, so a stored
    # solution only moves to a slot at or before its own.
    for slot in 1:survivors
        m = order[slot]
        archive.kept[m] = true
        if m <= stored
            slot == m && continue
            archive.genotypes[slot] = archive.genotypes[m]
            archive.phenotypes[slot] = archive.phenotypes[m]
        else
            archive.genotypes[slot] = xs[m-stored]
            store_phenotype!(archive.phenotypes, slot, stored_phenotype(P, evs[m-stored].phenotype))
        end
        archive.fitness[slot] = archive.merged_fitness[m]
        archive.descriptors[slot] = archive.merged_descriptors[m]
    end
    archive.n = survivors
    refresh_totals!(archive)
    return stored
end

# Mean distance from merged solution `i` to its `k` nearest other solutions with fitness
# at least as high, or `Inf` if there is none.
function dominated_novelty(archive::DNSArchive, i, merged)
    nearest = archive.nearest
    k = length(nearest)
    count = 0
    fitness = archive.merged_fitness[i]
    point = archive.merged_descriptors[i]
    for j in 1:merged
        (j == i || archive.merged_fitness[j] < fitness) && continue
        distance = sqrt(sum(abs2, point - archive.merged_descriptors[j]))
        count == k && distance >= nearest[k] && continue
        # insert into the sorted list of the nearest, dropping the farthest if full
        position = count < k ? (count += 1) : k
        while position > 1 && nearest[position-1] > distance
            nearest[position] = nearest[position-1]
            position -= 1
        end
        nearest[position] = distance
    end
    count == 0 && return Inf
    total = 0.0
    for c in 1:count
        total += nearest[c]
    end
    return total / count
end

function refresh_totals!(archive::DNSArchive{G,F}) where {G,F}
    archive.fitness_sum = 0.0
    archive.fitness_sum_error = 0.0
    archive.fitness_max = zero(F)
    for i in 1:archive.n
        _add_to_sum!(archive, Float64(archive.fitness[i]))
        archive.fitness_max = i == 1 ? archive.fitness[i] : max(archive.fitness_max, archive.fitness[i])
    end
    return nothing
end

qd_score(archive::DNSArchive; offset=0) = _fitness_sum(archive) + length(archive) * offset

function max_fitness(archive::DNSArchive)
    isempty_archive(archive, "max_fitness")
    return archive.fitness_max
end

function mean_fitness(archive::DNSArchive)
    isempty_archive(archive, "mean_fitness")
    return _fitness_sum(archive) / length(archive)
end

"""
    sample_elites(rng, archive::DNSArchive, n) -> Vector{G}

`n` genotypes drawn with replacement, uniformly over the population. Throws an
`ArgumentError` if the archive is empty.
"""
function sample_elites(rng::AbstractRNG, archive::DNSArchive{G}, n::Integer) where {G}
    archive.n == 0 && throw(ArgumentError("cannot sample elites from an empty archive"))
    parents = Vector{G}(undef, n)
    for j in 1:n
        parents[j] = archive.genotypes[rand(rng, 1:archive.n)]
    end
    return parents
end

function elites(archive::DNSArchive)
    return ((archive.genotypes[i], Evaluation(archive.fitness[i], archive.descriptors[i], archive.phenotypes[i]))
            for i in 1:archive.n)
end
