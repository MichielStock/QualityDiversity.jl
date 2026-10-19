"""
    DuplicatePolicy

Supertype of the rules that decide whether a [`ParetoGridArchive`](@ref) takes a solution
whose objectives equal those of a solution in the front of its cell:
[`RejectDuplicates`](@ref) and [`KeepDuplicates`](@ref).
"""
abstract type DuplicatePolicy end

"""
    RejectDuplicates()

Reject a solution whose objectives equal those of a solution in the front of its cell, so
that a front holds every objective vector once. The default of [`ParetoGridArchive`](@ref).
The MOME paper and QDax's `MOMERepertoire` keep both; see [`KeepDuplicates`](@ref).
"""
struct RejectDuplicates <: DuplicatePolicy end

"""
    KeepDuplicates()

Take a solution whose objectives equal those of a solution in the front of its cell, as the
MOME paper and QDax's `MOMERepertoire` do: only a solution that is at least as good in every
objective and better in one keeps it out. Different genotypes with equal objectives then all
stay, in slots of the front that add no hypervolume.
"""
struct KeepDuplicates <: DuplicatePolicy end

"""
    FullFrontPolicy

Supertype of the rules that decide what a [`ParetoGridArchive`](@ref) does with a solution
for a full front, one that holds `max_front` solutions of which the new one beats none in
every objective: [`RejectNewcomer`](@ref) and [`ReplaceCrowded`](@ref).
"""
abstract type FullFrontPolicy end

"""
    RejectNewcomer()

Reject the solution, so that a full front changes only when a new solution beats one of its
solutions in every objective. The default of [`ParetoGridArchive`](@ref), as in QDax's
`MOMERepertoire`; the text of the MOME paper replaces a random solution of the front instead.
"""
struct RejectNewcomer <: FullFrontPolicy end

"""
    ReplaceCrowded()

Of the solutions of the front and the new one, remove the one with the smallest crowding
distance, so that the front stays spread out, as MOME-PGX does (Janmohamed, Pierrot and
Cully, 2023, "Improving the Data Efficiency of Multi-Objective Quality-Diversity through
Gradient Assistance and Crowding Exploration").

The crowding distance is that of NSGA-II: for each objective, the gap between the two
neighbours of a solution in that objective divided by the range of the objective over these
solutions, summed over the objectives. The solutions with the smallest and the largest value
of an objective have an infinite distance, so the ends of the front are never removed. An
objective in which all of them are equal adds nothing. On a tie the new solution is removed,
and otherwise the one stored first among those tied.

When the new solution stays, its result is `Improved` and its `improvement` is the change in
the hypervolume of the cell, which is negative when the removed solution covered more.
"""
struct ReplaceCrowded <: FullFrontPolicy end

"""
    ParetoGridArchive{G}(dims, bounds, reference_point; max_front=50, outside=ThrowOutside(),
                         duplicates=RejectDuplicates(), full_front=RejectNewcomer(),
                         phenotype=Nothing)
    ParetoGridArchive(prob::AbstractQDProblem; dims, reference_point, max_front=50,
                      bounds=descriptor_bounds(prob), genotype=Vector{Float64},
                      outside=ThrowOutside(), duplicates=RejectDuplicates(),
                      full_front=RejectNewcomer(), phenotype=Nothing)

Archive for several objectives, the one of MOME (Pierrot et al., 2022): the cells are those
of a [`GridArchive`](@ref), and every cell keeps a Pareto front, the solutions of the cell
that no other solution of the cell beats in every objective, instead of one elite.

`dims` is the number of cells along each descriptor dimension and `bounds` the box that
they cover, as for a [`GridArchive`](@ref). The fitness of a solution is a vector of `M`
objectives, all maximized, and `reference_point`, a vector of `M` numbers, is the corner
from which hypervolumes are measured. It should lie below the objective values that matter,
since a solution contributes hypervolume only where it is larger than the reference point in
every objective. A cell keeps at most `max_front` solutions. The archive stores its
fitness values as `SVector{M,Float64}`. `phenotype` is the type of the phenotypes the
archive stores, `Nothing` for none; see [`Evaluation`](@ref).

[`add!`](@ref QualityDiversity.add!) puts a solution into the front of its cell unless a
solution there is at least as good in every objective and better in one; it removes the
solutions that the new one beats in every objective. `duplicates` decides about a solution
whose objectives equal those of a stored one: [`RejectDuplicates`](@ref) (the default) or
[`KeepDuplicates`](@ref). `full_front` decides about a solution that beats none of a full
front: [`RejectNewcomer`](@ref) (the default, as in QDax) rejects it and
[`ReplaceCrowded`](@ref) removes the most crowded of the front and the new solution. The
result is `NewCell` for a cell that was empty and `Improved` otherwise, and its
`improvement` is the hypervolume that the cell gains, a `Float64`; a rejected
solution has an `improvement` of zero. The results of [`add_batch!`](@ref
QualityDiversity.add_batch!) are those of each solution against the archive as it was
before the batch, and the solutions are then added in order.

`length(archive)` is the number of occupied cells, as for the other archives, and
[`num_solutions`](@ref) the number of solutions. [`elites`](@ref QualityDiversity.elites)
iterates over all of them. [`sample_elites`](@ref QualityDiversity.sample_elites) draws an
occupied cell uniformly and then a solution of its front uniformly.

# Metrics

[`qd_score`](@ref) is the MOQD score, the sum over the cells of the hypervolume of their
fronts; it takes no offset. [`coverage`](@ref) is the fraction of cells that are occupied,
[`max_hypervolume`](@ref) the largest hypervolume of a cell and
[`global_hypervolume`](@ref) the hypervolume of the Pareto front of all the solutions.
`max_fitness` and `mean_fitness` are not defined for several objectives.

The hypervolume of a front is recomputed whenever a cell changes. It takes no allocation
for two objectives; for more it does allocate.

# Example

```jldoctest
julia> using QualityDiversity, StaticArrays

julia> archive = ParetoGridArchive{Vector{Float64}}((2, 2), ((0, 1), (0, 1)), [0.0, 0.0]);

julia> QualityDiversity.add!(archive, [0.1], Evaluation(SVector(1.0, 3.0), SVector(0.2, 0.2))).status
NewCell::AddStatus = 2

julia> QualityDiversity.add!(archive, [0.2], Evaluation(SVector(3.0, 1.0), SVector(0.2, 0.2))).improvement
2.0

julia> length(archive), num_solutions(archive), qd_score(archive)
(1, 2, 5.0)
```
"""
mutable struct ParetoGridArchive{G,M,N,O<:OutsidePolicy,P,U<:DuplicatePolicy,R<:FullFrontPolicy} <: AbstractArchive{G,SVector{M,Float64},SVector{N,Float64},P}
    const dims::NTuple{N,Int}
    const lower::SVector{N,Float64}
    const upper::SVector{N,Float64}
    const outside::O
    const duplicates::U
    const full_front::R
    const max_front::Int
    const reference_point::SVector{M,Float64}
    # Slot `(i - 1) * max_front + k` holds the `k`-th solution of the front of cell `i`,
    # for `k` up to `counts[i]`.
    const counts::Vector{Int}
    const genotypes::Vector{G}
    const phenotypes::Vector{P}
    const fitness::Vector{SVector{M,Float64}}
    const descriptors::Vector{SVector{N,Float64}}
    # Hypervolume of the front of every cell, and the cells that are occupied.
    const volumes::Vector{Float64}
    const occupied_idx::Vector{Int}
    # Running total of `volumes` over the cells, and the rounding error of the sum.
    volume_sum::Float64
    volume_sum_error::Float64
    num_solutions::Int
    # Workspace for the hypervolume and the crowding distances of one front.
    const scratch::Vector{SVector{M,Float64}}
    const crowding::Vector{Float64}
    const order::Vector{Int}

    function ParetoGridArchive{G,M,N,O,P,U,R}(dims, lower, upper, reference_point, max_front, outside,
                                              duplicates, full_front) where {G,M,N,O<:OutsidePolicy,P,U<:DuplicatePolicy,R<:FullFrontPolicy}
        for i in 1:N
            dims[i] >= 1 ||
                throw(ArgumentError("dims[$i] is $(dims[i]) but every dimension needs at least one cell"))
            check_bound(i, lower[i], upper[i])
        end
        M >= 2 || throw(ArgumentError("the reference point has $M entries but several objectives need at least 2"))
        all(isfinite, reference_point) || throw(ArgumentError("the reference point must be finite"))
        max_front >= 1 || throw(ArgumentError("max_front must be at least 1, got $max_front"))
        ncells = prod(dims)
        slots = ncells * max_front
        return new{G,M,N,O,P,U,R}(dims, lower, upper, outside, duplicates, full_front, max_front, reference_point,
                              zeros(Int, ncells),
                              Vector{G}(undef, slots),
                              Vector{P}(undef, slots),
                              Vector{SVector{M,Float64}}(undef, slots),
                              Vector{SVector{N,Float64}}(undef, slots),
                              zeros(ncells), Int[], 0.0, 0.0, 0,
                              Vector{SVector{M,Float64}}(undef, max_front + 1),
                              Vector{Float64}(undef, max_front + 1), Vector{Int}(undef, max_front + 1))
    end
end

function ParetoGridArchive{G}(dims::Tuple, bounds::Tuple, reference_point::AbstractVector{<:Real};
                              max_front::Integer=50, outside::OutsidePolicy=ThrowOutside(),
                              duplicates::DuplicatePolicy=RejectDuplicates(),
                              full_front::FullFrontPolicy=RejectNewcomer(), phenotype::Type=Nothing) where {G}
    N = length(dims)
    length(bounds) == N ||
        throw(ArgumentError("dims has $N entries but bounds has $(length(bounds))"))
    N >= 1 || throw(ArgumentError("dims must have at least one entry"))
    M = length(reference_point)
    lower = SVector{N,Float64}(map(b -> Float64(b[1]), bounds))
    upper = SVector{N,Float64}(map(b -> Float64(b[2]), bounds))
    return ParetoGridArchive{G,M,N,typeof(outside),phenotype,typeof(duplicates),typeof(full_front)}(
        map(Int, dims), lower, upper, SVector{M,Float64}(reference_point), Int(max_front), outside,
        duplicates, full_front)
end

function ParetoGridArchive(dims::Tuple, bounds::Tuple, reference_point::AbstractVector{<:Real}; kwargs...)
    return ParetoGridArchive{Vector{Float64}}(dims, bounds, reference_point; kwargs...)
end

function ParetoGridArchive(prob::AbstractQDProblem; dims::Tuple, reference_point::AbstractVector{<:Real},
                           max_front::Integer=50, bounds::Tuple=descriptor_bounds(prob),
                           genotype::Type=Vector{Float64}, outside::OutsidePolicy=ThrowOutside(),
                           duplicates::DuplicatePolicy=RejectDuplicates(),
                           full_front::FullFrontPolicy=RejectNewcomer(), phenotype::Type=Nothing)
    return ParetoGridArchive{genotype}(dims, bounds, reference_point; max_front, outside, duplicates,
                                       full_front, phenotype)
end

Base.length(archive::ParetoGridArchive) = length(archive.occupied_idx)

capacity(archive::ParetoGridArchive) = length(archive.counts)

"""
    num_solutions(archive)

Number of solutions in `archive`. For a [`ParetoGridArchive`](@ref), which keeps a front in
every cell, it is larger than `length(archive)`, the number of occupied cells.
"""
num_solutions(archive::ParetoGridArchive) = archive.num_solutions

descriptor_bounds(archive::ParetoGridArchive) = Tuple(zip(archive.lower, archive.upper))

reference_point(archive::ParetoGridArchive) = archive.reference_point

"""
    cell_index(archive::ParetoGridArchive, d)

Linear index of the cell that contains descriptor `d`, as for a
[`GridArchive`](@ref): `0` for a descriptor that the `outside` policy rejects.
"""
function cell_index(archive::ParetoGridArchive, d::AbstractVector{<:Real})
    return grid_cell(archive.dims, archive.lower, archive.upper, archive.outside, d)
end

stored_descriptor(::ParetoGridArchive{G,M,N}, d) where {G,M,N} = SVector{N,Float64}(d)

# `a` is at least as large as `b` in every objective.
weakly_dominates(a::SVector{M}, b::SVector{M}) where {M} = all(a .>= b)

# `a` is at least as large as `b` in every objective and larger in one.
dominates(a::SVector{M}, b::SVector{M}) where {M} = all(a .>= b) && any(a .> b)

function objectives(archive::ParetoGridArchive{G,M}, ev::Evaluation) where {G,M}
    fitness = ev.fitness
    fitness isa AbstractVector ||
        throw(ArgumentError("the fitness of a solution must be a vector of $M objectives for a ParetoGridArchive, got a $(typeof(fitness))"))
    length(fitness) == M ||
        throw(ArgumentError("fitness has $(length(fitness)) objectives but the archive has $M"))
    f = SVector{M,Float64}(fitness)
    all(isfinite, f) || throw(ArgumentError("every objective must be finite, got $fitness"))
    return f
end

# Hypervolume of the `m` points in `scratch`.
function scratch_volume(archive::ParetoGridArchive{G,2}, m) where {G}
    scratch = archive.scratch
    ref = archive.reference_point
    # Insertion sort by decreasing first objective; the points are few.
    for a in 2:m
        point = scratch[a]
        b = a - 1
        while b >= 1 && scratch[b][1] < point[1]
            scratch[b+1] = scratch[b]
            b -= 1
        end
        scratch[b+1] = point
    end
    area = 0.0
    height = ref[2]
    for a in 1:m
        point = scratch[a]
        point[1] > ref[1] && point[2] > ref[2] || continue
        height = max(height, point[2])
        next = a < m ? max(scratch[a+1][1], ref[1]) : ref[1]
        area += (point[1] - next) * (height - ref[2])
    end
    return area
end

function scratch_volume(archive::ParetoGridArchive, m)
    return hypervolume((archive.scratch[k] for k in 1:m), archive.reference_point)
end

# A stored solution with objectives `g` keeps out a new one with objectives `f`.
blocks(::RejectDuplicates, g, f) = weakly_dominates(g, f)
blocks(::KeepDuplicates, g, f) = dominates(g, f)

# What adding a solution with objectives `f` to cell `i` does, without changing the
# archive: the status and the hypervolume that the cell gains, and the position in the
# front of the solution removed to make room for it (`0` for none).
function front_outcome(archive::ParetoGridArchive, i, f)
    n = archive.counts[i]
    base = (i - 1) * archive.max_front
    kept = 0
    for k in 1:n
        g = archive.fitness[base+k]
        blocks(archive.duplicates, g, f) && return AddResult(Rejected, 0.0), 0
        if !dominates(f, g)
            kept += 1
            archive.scratch[kept] = g
        end
    end
    archive.scratch[kept+1] = f
    remove = 0
    if kept == archive.max_front
        # The front is full and the new solution beats none of it, so `scratch[k]` holds
        # the objectives of its `k`-th solution.
        remove = make_room(archive.full_front, archive, kept + 1)
        remove == kept + 1 && return AddResult(Rejected, 0.0), 0
        archive.scratch[remove] = f
    end
    gain = scratch_volume(archive, remove == 0 ? kept + 1 : kept) - archive.volumes[i]
    return AddResult(n == 0 ? NewCell : Improved, gain), remove
end

# Which of the `m` points in `scratch` leaves a full front; the last of them is the new
# solution.
make_room(::RejectNewcomer, archive, m) = m

function make_room(::ReplaceCrowded, archive::ParetoGridArchive, m)
    crowding_distances!(archive, m)
    distances = archive.crowding
    remove = m
    for k in 1:m-1
        distances[k] < distances[remove] && (remove = k)
    end
    return remove
end

# NSGA-II crowding distance of each of the `m` points in `scratch`, into `crowding`.
function crowding_distances!(archive::ParetoGridArchive{G,M}, m) where {G,M}
    points, distances, order = archive.scratch, archive.crowding, archive.order
    for k in 1:m
        distances[k] = 0.0
    end
    for j in 1:M
        # Insertion sort of the points by objective `j`, stable, so the ends of the order
        # do not depend on how the sort breaks ties.
        for a in 1:m
            value = points[a][j]
            b = a - 1
            while b >= 1 && points[order[b]][j] > value
                order[b+1] = order[b]
                b -= 1
            end
            order[b+1] = a
        end
        range = points[order[m]][j] - points[order[1]][j]
        range > 0 || continue
        distances[order[1]] = Inf
        distances[order[m]] = Inf
        for a in 2:m-1
            distances[order[a]] += (points[order[a+1]][j] - points[order[a-1]][j]) / range
        end
    end
    return distances
end

# Puts a solution that `front_outcome` accepted into cell `i`, removing the solution at
# position `remove` of the front (none for `0`), and returns the change in the
# hypervolume of the cell.
function insert_front!(archive::ParetoGridArchive, i, x, f, descriptor, phenotype, remove)
    n = archive.counts[i]
    base = (i - 1) * archive.max_front
    kept = 0
    for k in 1:n
        g = archive.fitness[base+k]
        (k == remove || dominates(f, g)) && continue
        kept += 1
        if kept != k
            archive.genotypes[base+kept] = archive.genotypes[base+k]
            move_phenotype!(archive.phenotypes, base + kept, base + k)
            archive.fitness[base+kept] = g
            archive.descriptors[base+kept] = archive.descriptors[base+k]
        end
        archive.scratch[kept] = g
    end
    archive.genotypes[base+kept+1] = x
    store_phenotype!(archive.phenotypes, base + kept + 1, phenotype)
    archive.fitness[base+kept+1] = f
    archive.descriptors[base+kept+1] = descriptor
    archive.scratch[kept+1] = f
    n == 0 && push!(archive.occupied_idx, i)
    archive.counts[i] = kept + 1
    archive.num_solutions += kept + 1 - n
    volume = scratch_volume(archive, kept + 1)
    change = volume - archive.volumes[i]
    archive.volumes[i] = volume
    add_to_volume!(archive, change)
    return change
end

# Neumaier's compensated summation of the running total.
function add_to_volume!(archive::ParetoGridArchive, x::Float64)
    total = archive.volume_sum
    updated = total + x
    archive.volume_sum_error += abs(total) >= abs(x) ? (total - updated) + x : (x - updated) + total
    archive.volume_sum = updated
    return nothing
end

"""
    add!(archive::ParetoGridArchive, x, ev::Evaluation) -> AddResult

Insert `x` into the front of the cell of `ev.descriptor`; see [`ParetoGridArchive`](@ref).
`ev.fitness` must be a vector with one entry for each objective of the archive.
"""
function add!(archive::ParetoGridArchive{G,M,N,O,P}, x, ev::Evaluation) where {G,M,N,O,P}
    phenotype = stored_phenotype(P, ev.phenotype)
    i = cell_index(archive, ev.descriptor)
    i == 0 && return AddResult(Rejected, 0.0)
    f = objectives(archive, ev)
    result, remove = front_outcome(archive, i, f)
    result.status == Rejected && return result
    insert_front!(archive, i, x, f, stored_descriptor(archive, ev.descriptor), phenotype, remove)
    return result
end

"""
    add_batch!(archive::ParetoGridArchive, xs, evs) -> Vector{AddResult}

Insert every solution in `xs` with its evaluation from `evs`. The status and
`improvement` of each solution are those it would have if it were added alone to the
archive as it was before the batch, so they do not depend on the other solutions in the
batch. The solutions are then offered in order, each against the front of its cell as it
is by then, so the archive afterwards is the one that adding them one at a time gives: a
solution that another solution of the batch beats is removed, and one reported `Rejected`
can still enter a front in which a solution before it in the batch made room.
"""
function add_batch!(archive::ParetoGridArchive{G,M,N,O,P}, xs::AbstractVector,
                    evs::AbstractVector) where {G,M,N,O,P}
    length(xs) == length(evs) ||
        throw(ArgumentError("got $(length(xs)) solutions but $(length(evs)) evaluations"))
    Base.require_one_based_indexing(xs, evs)
    n = length(xs)
    results = Vector{AddResult{Float64}}(undef, n)
    cells = Vector{Int}(undef, n)
    fitness = Vector{eltype(archive.fitness)}(undef, n)
    for k in 1:n
        i = cell_index(archive, evs[k].descriptor)
        stored_phenotype(P, evs[k].phenotype)    # throws before the archive changes
        cells[k] = i
        if i == 0
            results[k] = AddResult(Rejected, 0.0)
        else
            fitness[k] = objectives(archive, evs[k])
            results[k] = first(front_outcome(archive, i, fitness[k]))
        end
    end
    for k in 1:n
        i = cells[k]
        i == 0 && continue
        # The front may have changed since the result: offered by now, as `add!` would.
        outcome, remove = front_outcome(archive, i, fitness[k])
        outcome.status == Rejected && continue
        insert_front!(archive, i, xs[k], fitness[k], stored_descriptor(archive, evs[k].descriptor),
                      stored_phenotype(P, evs[k].phenotype), remove)
    end
    return results
end

"""
    sample_elites(rng, archive::ParetoGridArchive, n) -> Vector{G}

`n` genotypes drawn with replacement: for each, an occupied cell uniformly and then a
solution of its front uniformly. Throws an `ArgumentError` if the archive is empty.
"""
function sample_elites(rng::AbstractRNG, archive::ParetoGridArchive{G}, n::Integer) where {G}
    isempty(archive.occupied_idx) &&
        throw(ArgumentError("cannot sample elites from an empty archive"))
    parents = Vector{G}(undef, n)
    for k in 1:n
        i = rand(rng, archive.occupied_idx)
        parents[k] = archive.genotypes[(i-1)*archive.max_front+rand(rng, 1:archive.counts[i])]
    end
    return parents
end

function elites(archive::ParetoGridArchive)
    L = archive.max_front
    return ((archive.genotypes[slot], Evaluation(archive.fitness[slot], archive.descriptors[slot], archive.phenotypes[slot]))
            for i in archive.occupied_idx for slot in (i-1)*L+1:(i-1)*L+archive.counts[i])
end

function Base.empty!(archive::ParetoGridArchive)
    fill!(archive.counts, 0)
    fill!(archive.volumes, 0.0)
    empty!(archive.occupied_idx)
    archive.volume_sum = 0.0
    archive.volume_sum_error = 0.0
    archive.num_solutions = 0
    return archive
end

function qd_score(archive::ParetoGridArchive; offset=0)
    offset == 0 ||
        throw(ArgumentError("the MOQD score of a ParetoGridArchive takes no offset: hypervolume is measured from the reference point"))
    return archive.volume_sum + archive.volume_sum_error
end

"""
    max_hypervolume(archive::ParetoGridArchive)

Largest hypervolume of the front of one cell. Throws an `ArgumentError` for an empty
archive.
"""
function max_hypervolume(archive::ParetoGridArchive)
    isempty_archive(archive, "max_hypervolume")
    return maximum(archive.volumes[i] for i in archive.occupied_idx)
end

"""
    global_hypervolume(archive::ParetoGridArchive)

Hypervolume of the Pareto front of all the solutions of the archive, from every cell. It
compares all solutions with each other, so it takes time quadratic in
[`num_solutions`](@ref). Zero for an empty archive.
"""
function global_hypervolume(archive::ParetoGridArchive)
    points = [ev.fitness for (_, ev) in elites(archive)]
    return hypervolume(points, archive.reference_point)
end

max_fitness(archive::ParetoGridArchive) =
    throw(ArgumentError("max_fitness is not defined for a ParetoGridArchive, which has several objectives: use max_hypervolume or max_scores"))

mean_fitness(archive::ParetoGridArchive) =
    throw(ArgumentError("mean_fitness is not defined for a ParetoGridArchive, which has several objectives"))

"""
    max_scores(archive::ParetoGridArchive) -> SVector

Highest value of every objective over all the solutions of the archive. Throws an
`ArgumentError` for an empty archive.
"""
function max_scores(archive::ParetoGridArchive{G,M}) where {G,M}
    isempty_archive(archive, "max_scores")
    best = SVector{M,Float64}(ntuple(_ -> -Inf, Val(M)))
    for (_, ev) in elites(archive)
        best = max.(best, ev.fitness)
    end
    return best
end

const ParetoMetricsRecord = @NamedTuple{n_evals::Int, qd_score::Float64, coverage::Float64,
                                        max_hypervolume::Float64, num_solutions::Int}

metrics_record_type(::Type{<:ParetoGridArchive}) = ParetoMetricsRecord

function metrics_record(archive::ParetoGridArchive, n_evals, offset)
    return (n_evals=n_evals, qd_score=qd_score(archive), coverage=Float64(coverage(archive)),
            max_hypervolume=length(archive) == 0 ? 0.0 : max_hypervolume(archive),
            num_solutions=num_solutions(archive))
end
