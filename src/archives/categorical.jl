"""
    CategoricalArchive{G,F}(categories; outside=ThrowOutside())
    CategoricalArchive{G}(categories; outside=ThrowOutside())
    CategoricalArchive(categories; outside=ThrowOutside())

Archive with one cell for each combination of discrete descriptor values, and
the highest-fitness solution found in each. Use it when a descriptor counts or
labels something, such as the number of rings in a molecule or a scaffold
class, so that only a few values occur and nothing lies between them.

`categories` is a tuple with one collection of allowed values per descriptor
dimension, for example `(0:5, [1, 2, 4])`. The values are real numbers; a label
that is not a number is encoded as one by the problem. Each collection must be
non-empty with finite, distinct values. The cells are ordered like those of a
[`GridArchive`](@ref): the values of each dimension in ascending order, and the
first dimension varying fastest. `G` is the genotype type, `Vector{Float64}` by
default, and `F<:Real` the fitness type, `Float64` by default.

A descriptor belongs to a cell if each component is exactly equal to one of the
values of its dimension. `outside` says what happens to a descriptor with a
component that is not (including `NaN`):

- `ThrowOutside()` (default): throw an `ArgumentError`.
- `RejectOutside()`: discard the solution; `add!` returns a `Rejected` result.

`ClampOutside()` has no meaning for categories and is an error. The problem's
`descriptor_bounds` need only have one pair per dimension, as `init` checks.

A solution is stored without copying, so the caller must not modify it
afterwards. The archive keeps the same running totals as
[`GridArchive`](@ref), so the metrics take constant time.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = CategoricalArchive((0:3, [10, 20]));

julia> length(archive), QualityDiversity.capacity(archive)
(0, 8)

julia> QualityDiversity.cell_index(archive, [2, 20])
7
```
"""
mutable struct CategoricalArchive{G,F<:Real,N,O<:Union{ThrowOutside,RejectOutside}} <: CellArchive{G,F,SVector{N,Float64}}
    # Ascending values of each dimension.
    const categories::NTuple{N,Vector{Float64}}
    const outside::O
    const occupied::BitVector
    const fitness::Vector{F}
    const descriptors::Vector{SVector{N,Float64}}
    const genotypes::Vector{G}
    const occupied_idx::Vector{Int}
    # Running totals, as in GridArchive.
    fitness_sum::Float64
    fitness_sum_error::Float64
    fitness_max::F

    function CategoricalArchive{G,F,N}(categories::NTuple{N,Vector{Float64}},
                                       outside::O) where {G,F<:Real,N,O<:Union{ThrowOutside,RejectOutside}}
        for i in 1:N
            values = categories[i]
            isempty(values) && throw(ArgumentError("categories[$i] is empty; every dimension needs at least one value"))
            all(isfinite, values) ||
                throw(ArgumentError("categories[$i] is $values but its values must be finite"))
            for j in 2:length(values)
                values[j] == values[j-1] &&
                    throw(ArgumentError("categories[$i] contains $(values[j]) more than once"))
            end
        end
        ncells = prod(length, categories)
        return new{G,F,N,O}(categories, outside,
                            falses(ncells),
                            Vector{F}(undef, ncells),
                            Vector{SVector{N,Float64}}(undef, ncells),
                            Vector{G}(undef, ncells),
                            Int[],
                            0.0, 0.0, zero(F))
    end
end

function CategoricalArchive{G,F}(categories::Tuple; outside::OutsidePolicy=ThrowOutside()) where {G,F<:Real}
    N = length(categories)
    N >= 1 || throw(ArgumentError("categories must have at least one entry"))
    sorted = ntuple(N) do i
        sort!(Float64[v for v in categories[i]])
    end
    return CategoricalArchive{G,F,N}(sorted, check_categorical_outside(outside))
end

check_categorical_outside(outside::Union{ThrowOutside,RejectOutside}) = outside
function check_categorical_outside(::ClampOutside)
    throw(ArgumentError("ClampOutside() has no meaning for a CategoricalArchive; use ThrowOutside() or RejectOutside()"))
end

CategoricalArchive{G}(categories::Tuple; kwargs...) where {G} =
    CategoricalArchive{G,Float64}(categories; kwargs...)

CategoricalArchive(categories::Tuple; kwargs...) =
    CategoricalArchive{Vector{Float64}}(categories; kwargs...)

# Position of `x` among `values`, or what the outside policy makes of a value
# that is not there.
function category_position(archive::CategoricalArchive, i, x)
    # The assertion keeps the element type known when the length of the tuple is not.
    values = archive.categories[i]::Vector{Float64}
    k = searchsortedfirst(values, x)
    k <= length(values) && values[k] == x && return k
    return unknown_category(archive.outside, i, x, values)
end

function unknown_category(::ThrowOutside, i, x, values)
    shown = length(values) <= 8 ? string(values) :
            string(first(values, 4), " … ", last(values, 2), " (", length(values), " values)")
    throw(ArgumentError("descriptor component $i is $x, which is not one of the categories $shown"))
end

unknown_category(::RejectOutside, i, x, values) = 0

"""
    cell_index(archive::CategoricalArchive, d)

Linear index of the cell whose categories equal the components of descriptor
`d`. Throws an `ArgumentError` if `d` has the wrong number of dimensions. What
happens to a component that is not a category, or is `NaN`, depends on the
archive's `outside` policy: `ThrowOutside()` throws and `RejectOutside()`
returns `0`, which is not a cell.
"""
function cell_index(archive::CategoricalArchive{G,F,N}, d::AbstractVector{<:Real}) where {G,F,N}
    length(d) == N ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $N"))
    index = 0
    stride = 1
    for i in 1:N
        k = category_position(archive, i, Float64(d[i]))
        k == 0 && return 0
        index += (k - 1) * stride
        stride *= length(archive.categories[i])
    end
    return index + 1
end
