"""
    OutsidePolicy

Supertype of the policies that tell a [`GridArchive`](@ref) what to do with a
descriptor outside its bounds: `ThrowOutside()`, `ClampOutside()` and `RejectOutside()`.
"""
abstract type OutsidePolicy end

"ThrowOutside an `ArgumentError` for a descriptor outside the bounds."
struct ThrowOutside <: OutsidePolicy end

"Use the nearest cell for a descriptor outside the bounds."
struct ClampOutside <: OutsidePolicy end

"Discard a solution whose descriptor is outside the bounds."
struct RejectOutside <: OutsidePolicy end

"""
    GridArchive{G,F}(dims, bounds; outside=ThrowOutside())
    GridArchive{G}(dims, bounds; outside=ThrowOutside())
    GridArchive(dims, bounds; outside=ThrowOutside())
    GridArchive(prob::AbstractQDProblem; dims, bounds=descriptor_bounds(prob),
                genotype=Vector{Float64}, fitness=Float64, outside=ThrowOutside())

Archive that divides descriptor space into a regular grid and keeps the
highest-fitness solution found in each cell.

`dims` is a tuple with the number of cells along each descriptor dimension.
`bounds` is a tuple with one `(lo, hi)` pair per dimension. `G` is the
genotype type, `Vector{Float64}` by default, and `F<:Real` the fitness type,
`Float64` by default. The last form takes the bounds from
[`descriptor_bounds(prob)`](@ref QualityDiversity.descriptor_bounds) unless
`bounds` is given, for example to zoom in on part of descriptor space.

Descriptors are `SVector{N,Float64}`. `outside` says what happens to a
descriptor that falls outside the bounds:

- `ThrowOutside()` (default): throw an `ArgumentError`.
- `ClampOutside()`: use the nearest cell. A `NaN` component still throws.
- `RejectOutside()`: discard the solution; `add!` returns a `Rejected` result.

A solution is stored without copying, so the caller must not modify it
afterwards.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = GridArchive((10, 10), ((0, 1), (0, 1)));

julia> length(archive), QualityDiversity.capacity(archive)
(0, 100)
```
"""
struct GridArchive{G,F<:Real,N,O<:OutsidePolicy} <: AbstractArchive{G,F,SVector{N,Float64}}
    dims::NTuple{N,Int}
    lower::SVector{N,Float64}
    upper::SVector{N,Float64}
    outside::O
    occupied::BitVector
    fitness::Vector{F}
    descriptors::Vector{SVector{N,Float64}}
    genotypes::Vector{G}
    occupied_idx::Vector{Int}

    function GridArchive{G,F,N}(dims, lower, upper, outside::O) where {G,F<:Real,N,O<:OutsidePolicy}
        for i in 1:N
            dims[i] >= 1 ||
                throw(ArgumentError("dims[$i] is $(dims[i]) but every dimension needs at least one cell"))
            (isfinite(lower[i]) && isfinite(upper[i]) && lower[i] < upper[i]) ||
                throw(ArgumentError("bounds[$i] is ($(lower[i]), $(upper[i])) but lo must be finite and less than hi"))
        end
        ncells = prod(dims)
        return new{G,F,N,O}(dims, lower, upper, outside,
                          falses(ncells),
                          Vector{F}(undef, ncells),
                          Vector{SVector{N,Float64}}(undef, ncells),
                          Vector{G}(undef, ncells),
                          Int[])
    end
end

function GridArchive{G,F}(dims::Tuple, bounds::Tuple;
                          outside::OutsidePolicy=ThrowOutside()) where {G,F<:Real}
    N = length(dims)
    length(bounds) == N ||
        throw(ArgumentError("dims has $N entries but bounds has $(length(bounds))"))
    N >= 1 || throw(ArgumentError("dims must have at least one entry"))
    lower = SVector{N,Float64}(map(b -> Float64(b[1]), bounds))
    upper = SVector{N,Float64}(map(b -> Float64(b[2]), bounds))
    return GridArchive{G,F,N}(map(Int, dims), lower, upper, outside)
end

function GridArchive{G}(dims::Tuple, bounds::Tuple; outside::OutsidePolicy=ThrowOutside()) where {G}
    return GridArchive{G,Float64}(dims, bounds; outside)
end

function GridArchive(dims::Tuple, bounds::Tuple; outside::OutsidePolicy=ThrowOutside())
    return GridArchive{Vector{Float64}}(dims, bounds; outside)
end

function GridArchive(prob::AbstractQDProblem; dims::Tuple,
                     bounds::Tuple=descriptor_bounds(prob),
                     genotype::Type=Vector{Float64}, fitness::Type{<:Real}=Float64,
                     outside::OutsidePolicy=ThrowOutside())
    return GridArchive{genotype,fitness}(dims, bounds; outside)
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
number of dimensions. What happens to a component outside the bounds or `NaN`
depends on the archive's `outside` policy: `ThrowOutside()` throws, `ClampOutside()` uses the
nearest cell (and throws for `NaN`), and `RejectOutside()` returns `0`, which is not a
cell.
"""
function cell_index(archive::GridArchive{G,F,N}, d::AbstractVector{<:Real}) where {G,F,N}
    length(d) == N ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $N"))
    index = 0
    stride = 1
    for i in 1:N
        lo = archive.lower[i]
        hi = archive.upper[i]
        x = d[i]
        if !(lo <= x <= hi)
            x = resolve_outside(archive.outside, i, x, lo, hi)
            isnan(x) && return 0
        end
        n = archive.dims[i]
        k = min(floor(Int, (x - lo) / (hi - lo) * n) + 1, n)
        index += (k - 1) * stride
        stride *= n
    end
    return index + 1
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
    add!(archive::GridArchive, x, ev::Evaluation) -> AddResult

Insert `x` into the cell of `ev.descriptor`. A descriptor outside the bounds is
handled by the archive's `outside` policy; `RejectOutside()` gives a `Rejected` result
with zero `improvement`. An empty cell takes any solution
(`NewCell`, with `improvement` equal to the fitness); an occupied cell takes it
only if the fitness is strictly higher (`Improved`, otherwise `Rejected`).
For an occupied cell, `improvement` is the fitness minus the occupant's.
"""
function add!(archive::GridArchive{G,F}, x,
              ev::Evaluation{<:Real,<:AbstractVector{<:Real}}) where {G,F}
    i = cell_index(archive, ev.descriptor)
    fitness = convert(F, ev.fitness)
    i == 0 && return AddResult(Rejected, zero(F))
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

function check_compatible(prob::AbstractQDProblem, archive::GridArchive{G,F,N}) where {G,F,N}
    n = length(descriptor_bounds(prob))
    n == N || throw(ArgumentError("the problem has $n descriptor dimensions but the archive has $N"))
    return nothing
end
