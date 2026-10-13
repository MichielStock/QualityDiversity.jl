"""
    GridArchive{G,F}(dims, bounds)
    GridArchive{G}(dims, bounds)
    GridArchive(dims, bounds)
    GridArchive(prob::AbstractQDProblem; dims, genotype=Vector{Float64}, fitness=Float64)

Archive that divides descriptor space into a regular grid and keeps the
highest-fitness solution found in each cell.

`dims` is a tuple with the number of cells along each descriptor dimension.
`bounds` is a tuple with one `(lo, hi)` pair per dimension. `G` is the
genotype type, `Vector{Float64}` by default, and `F<:Real` the fitness type,
`Float64` by default. The last form takes the bounds from
[`descriptor_bounds(prob)`](@ref QualityDiversity.descriptor_bounds).

Descriptors are `SVector{N,Float64}`. A descriptor outside the bounds is an
error, not clamped. A solution is stored without copying, so the caller must
not modify it afterwards.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = GridArchive((10, 10), ((0, 1), (0, 1)));

julia> length(archive), QualityDiversity.capacity(archive)
(0, 100)
```
"""
struct GridArchive{G,F<:Real,N} <: AbstractArchive{G,F,SVector{N,Float64}}
    dims::NTuple{N,Int}
    lower::SVector{N,Float64}
    upper::SVector{N,Float64}
    occupied::BitVector
    fitness::Vector{F}
    descriptors::Vector{SVector{N,Float64}}
    genotypes::Vector{G}
    occupied_idx::Vector{Int}

    function GridArchive{G,F,N}(dims, lower, upper) where {G,F<:Real,N}
        for i in 1:N
            dims[i] >= 1 ||
                throw(ArgumentError("dims[$i] is $(dims[i]) but every dimension needs at least one cell"))
            (isfinite(lower[i]) && isfinite(upper[i]) && lower[i] < upper[i]) ||
                throw(ArgumentError("bounds[$i] is ($(lower[i]), $(upper[i])) but lo must be finite and less than hi"))
        end
        ncells = prod(dims)
        return new{G,F,N}(dims, lower, upper,
                          falses(ncells),
                          Vector{F}(undef, ncells),
                          Vector{SVector{N,Float64}}(undef, ncells),
                          Vector{G}(undef, ncells),
                          Int[])
    end
end

function GridArchive{G,F}(dims::Tuple, bounds::Tuple) where {G,F<:Real}
    N = length(dims)
    length(bounds) == N ||
        throw(ArgumentError("dims has $N entries but bounds has $(length(bounds))"))
    N >= 1 || throw(ArgumentError("dims must have at least one entry"))
    lower = SVector{N,Float64}(map(b -> Float64(b[1]), bounds))
    upper = SVector{N,Float64}(map(b -> Float64(b[2]), bounds))
    return GridArchive{G,F,N}(map(Int, dims), lower, upper)
end

GridArchive{G}(dims::Tuple, bounds::Tuple) where {G} = GridArchive{G,Float64}(dims, bounds)

GridArchive(dims::Tuple, bounds::Tuple) = GridArchive{Vector{Float64}}(dims, bounds)

function GridArchive(prob::AbstractQDProblem; dims::Tuple,
                     genotype::Type=Vector{Float64}, fitness::Type{<:Real}=Float64)
    return GridArchive{genotype,fitness}(dims, descriptor_bounds(prob))
end

Base.length(archive::GridArchive) = length(archive.occupied_idx)

capacity(archive::GridArchive) = length(archive.occupied)

function Base.empty!(archive::GridArchive)
    fill!(archive.occupied, false)
    empty!(archive.occupied_idx)
    return archive
end

"""
    cell_index(archive::GridArchive, d)

Linear index of the cell that contains descriptor `d`. A value on the upper
bound belongs to the last cell. Throws an `ArgumentError` if `d` has the wrong
number of dimensions, or a component is outside the bounds or `NaN`.
"""
function cell_index(archive::GridArchive{G,F,N}, d::AbstractVector{<:Real}) where {G,F,N}
    length(d) == N ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $N"))
    index = 0
    stride = 1
    for i in 1:N
        x = d[i]
        lo = archive.lower[i]
        hi = archive.upper[i]
        (lo <= x <= hi) ||
            throw(ArgumentError("descriptor component $i is $x, outside the archive bounds [$lo, $hi]"))
        n = archive.dims[i]
        k = min(floor(Int, (x - lo) / (hi - lo) * n) + 1, n)
        index += (k - 1) * stride
        stride *= n
    end
    return index + 1
end

"""
    add!(archive::GridArchive, x, ev::Evaluation) -> AddResult

Insert `x` into the cell of `ev.descriptor`. An empty cell takes any solution
(`NewCell`, with `improvement` equal to the fitness); an occupied cell takes it
only if the fitness is strictly higher (`Improved`, otherwise `Rejected`).
For an occupied cell, `improvement` is the fitness minus the occupant's.
"""
function add!(archive::GridArchive{G,F}, x,
              ev::Evaluation{<:Real,<:AbstractVector{<:Real}}) where {G,F}
    i = cell_index(archive, ev.descriptor)
    fitness = convert(F, ev.fitness)
    if !archive.occupied[i]
        archive.occupied[i] = true
        push!(archive.occupied_idx, i)
        _store!(archive, i, x, fitness, ev.descriptor)
        return AddResult(NewCell, fitness)
    end
    improvement = fitness - archive.fitness[i]
    if improvement > 0
        _store!(archive, i, x, fitness, ev.descriptor)
        return AddResult(Improved, improvement)
    end
    return AddResult(Rejected, improvement)
end

function _store!(archive::GridArchive, i, x, fitness, descriptor)
    archive.genotypes[i] = x
    archive.fitness[i] = fitness
    archive.descriptors[i] = descriptor
    return nothing
end

"""
    sample_elites(rng, archive::GridArchive, n) -> Vector{G}

`n` genotypes drawn with replacement, uniformly over the occupied cells.
Throws an `ArgumentError` if the archive is empty.
"""
function sample_elites(rng::AbstractRNG, archive::GridArchive{G}, n::Integer) where {G}
    isempty(archive.occupied_idx) &&
        throw(ArgumentError("cannot sample elites from an empty archive"))
    parents = Vector{G}(undef, n)
    for k in 1:n
        parents[k] = archive.genotypes[rand(rng, archive.occupied_idx)]
    end
    return parents
end

function elites(archive::GridArchive)
    return ((archive.genotypes[i], Evaluation(archive.fitness[i], archive.descriptors[i]))
            for i in archive.occupied_idx)
end
