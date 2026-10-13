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

# Metrics

`coverage`, `qd_score`, `max_fitness` and `mean_fitness` take constant time:
the archive keeps the number of elites, the sum of their fitness and their
maximum fitness up to date inside `add!`, which already knows what changed.
A cell is only replaced by a strictly better solution, so the maximum never
decreases and a running maximum is exact. The sum changes by
`new fitness - old fitness` for an improved cell and by the fitness for a new
cell. It is accumulated in `Float64` with compensated (Neumaier) summation, so
the rounding error stays within about one rounding of the exact sum however
many updates there have been; a plain running sum drifts by hundreds of units
in the last place after millions of updates. The value agrees with a fresh pass
over the elites, which the generic metrics for any [`AbstractArchive`](@ref)
perform, to within that rounding.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = GridArchive((10, 10), ((0, 1), (0, 1)));

julia> length(archive), QualityDiversity.capacity(archive)
(0, 100)
```
"""
mutable struct GridArchive{G,F<:Real,N,O<:OutsidePolicy} <: CellArchive{G,F,SVector{N,Float64}}
    const dims::NTuple{N,Int}
    const lower::SVector{N,Float64}
    const upper::SVector{N,Float64}
    const outside::O
    const occupied::BitVector
    const fitness::Vector{F}
    const descriptors::Vector{SVector{N,Float64}}
    const genotypes::Vector{G}
    const occupied_idx::Vector{Int}
    # Running totals over the occupied cells, updated only by `add!` and reset
    # by `empty!`. `fitness_sum + fitness_sum_error` is the sum of the fitness
    # values; the second term collects the rounding error of the first.
    # `fitness_max` is meaningless while the archive is empty.
    fitness_sum::Float64
    fitness_sum_error::Float64
    fitness_max::F

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
                          Int[],
                          0.0, 0.0, zero(F))
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
