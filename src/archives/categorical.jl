"""
    CategoricalArchive{G,F}(categories; outside=ThrowOutside())
    CategoricalArchive{G}(categories; outside=ThrowOutside())
    CategoricalArchive(categories; outside=ThrowOutside())
    CategoricalArchive(prob::AbstractQDProblem; genotype=Vector{Float64},
                       fitness=Float64, outside=ThrowOutside())

Archive with one cell for each combination of discrete descriptor values, and
the highest-fitness solution found in each. Use it when a descriptor counts or
labels something, such as the number of rings in a molecule or a scaffold
class, so that only a few values occur and nothing lies between them.

`categories` is a tuple with one collection of allowed values per descriptor
dimension, for example `(0:5, [:alkane, :aromatic, :ring])`. A value can be
anything that has `hash` and `isequal`: a number, a `Symbol`, a `String`, a
`Bool`, or a tuple. Each collection must be non-empty and have distinct values,
and a number must be finite. The last form takes the categories from
[`descriptor_categories(prob)`](@ref QualityDiversity.descriptor_categories).

A descriptor is a tuple with one value per dimension, such as `(3, :ring)`. The
descriptor type of the archive is the tuple of the element types of
`categories`, here `Tuple{Int,Symbol}`, so the archive is concretely typed. A
vector with one entry per dimension, such as the `SVector` of a problem with
numeric categories, is accepted as well. Values compare with `isequal`, except
that `-0.0` equals `0`, so `2.0` matches the category `2`. A stored descriptor
holds the category values.

The cells are numbered like those of a [`GridArchive`](@ref), with the first
dimension varying fastest and the values of each dimension in the order given.
`G` is the genotype type, `Vector{Float64}` by default, and `F<:Real` the
fitness type, `Float64` by default. `outside` says what happens to a descriptor
with a component that is not a category:

- `ThrowOutside()` (default): throw an `ArgumentError`.
- `RejectOutside()`: discard the solution; `add!` returns a `Rejected` result.

`ClampOutside()` has no meaning for categories and is an error.

A category is found with a hash lookup, so a lookup takes constant time and does
not allocate however many categories there are. A solution is stored without
copying, so the caller must not modify it afterwards. The archive keeps the same
running totals as [`GridArchive`](@ref), so the metrics take constant time.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = CategoricalArchive((0:5, [:alkane, :aromatic, :ring]));

julia> length(archive), QualityDiversity.capacity(archive)
(0, 18)

julia> QualityDiversity.cell_index(archive, (3, :ring))
16
```
"""
mutable struct CategoricalArchive{G,F<:Real,D<:Tuple,C<:Tuple,L<:Tuple,O<:Union{ThrowOutside,RejectOutside}} <: CellArchive{G,F,D}
    # The allowed values of each dimension, in the order given.
    const categories::C
    # For each dimension, the position of each value in `categories`.
    const lookups::L
    const outside::O
    const occupied::BitVector
    const fitness::Vector{F}
    const descriptors::Vector{D}
    const genotypes::Vector{G}
    const occupied_idx::Vector{Int}
    # Running totals, as in GridArchive.
    fitness_sum::Float64
    fitness_sum_error::Float64
    fitness_max::F

    function CategoricalArchive{G,F,D}(categories::C, lookups::L,
                                       outside::O) where {G,F<:Real,D<:Tuple,C<:Tuple,L<:Tuple,
                                                          O<:Union{ThrowOutside,RejectOutside}}
        ncells = prod(length, categories)
        return new{G,F,D,C,L,O}(categories, lookups, outside,
                                falses(ncells),
                                Vector{F}(undef, ncells),
                                Vector{D}(undef, ncells),
                                Vector{G}(undef, ncells),
                                Int[],
                                0.0, 0.0, zero(F))
    end
end

function CategoricalArchive{G,F}(categories::Tuple; outside::OutsidePolicy=ThrowOutside()) where {G,F<:Real}
    isempty(categories) && throw(ArgumentError("categories must have at least one entry"))
    values = ntuple(i -> collect(categories[i]), length(categories))
    lookups = ntuple(i -> category_lookup(i, values[i]), length(categories))
    D = Tuple{map(eltype, values)...}
    return CategoricalArchive{G,F,D}(values, lookups, check_categorical_outside(outside))
end

CategoricalArchive{G}(categories::Tuple; kwargs...) where {G} =
    CategoricalArchive{G,Float64}(categories; kwargs...)

CategoricalArchive(categories::Tuple; kwargs...) =
    CategoricalArchive{Vector{Float64}}(categories; kwargs...)

function CategoricalArchive(prob::AbstractQDProblem; genotype::Type=Vector{Float64},
                           fitness::Type{<:Real}=Float64, outside::OutsidePolicy=ThrowOutside())
    hasmethod(descriptor_categories, Tuple{typeof(prob)}) ||
        throw(ArgumentError("$(typeof(prob)) defines no descriptor_categories; pass the categories to CategoricalArchive directly"))
    return CategoricalArchive{genotype,fitness}(descriptor_categories(prob); outside)
end

check_categorical_outside(outside::Union{ThrowOutside,RejectOutside}) = outside
function check_categorical_outside(::ClampOutside)
    throw(ArgumentError("ClampOutside() has no meaning for a CategoricalArchive; use ThrowOutside() or RejectOutside()"))
end

# Position of each of `values` in it, after checking that there is at least one
# value and that no value, or a non-finite number, occurs.
function category_lookup(i, values::AbstractVector{T}) where {T}
    isempty(values) && throw(ArgumentError("categories[$i] is empty; every dimension needs at least one value"))
    lookup = Dict{T,Int}()
    sizehint!(lookup, length(values))
    for (k, v) in enumerate(values)
        v isa Number && !isfinite(v) &&
            throw(ArgumentError("categories[$i] is $values but its values must be finite"))
        haskey(lookup, v) && throw(ArgumentError("categories[$i] contains $v more than once"))
        lookup[lookup_key(v)] = k
    end
    return lookup
end

# `isequal` tells `-0.0` from `0.0` and `0`; adding a zero maps `-0.0` to `0.0`.
lookup_key(x::AbstractFloat) = x + zero(x)
lookup_key(x) = x

function unknown_category(::ThrowOutside, i, x, values)
    shown = length(values) <= 8 ? string(values) :
            string(first(values, 4), " … ", last(values, 2), " (", length(values), " values)")
    throw(ArgumentError("descriptor component $i is $x, which is not one of the categories $shown"))
end

unknown_category(::RejectOutside, i, x, values) = 0

# Linear index of the cell of `d`, accumulated one dimension at a time; the
# recursion over the tuples is unrolled when the method is compiled, so each
# dimension is looked up with its own concrete key type.
cell_position(outside, ::Tuple{}, ::Tuple{}, ::Tuple{}, i, stride, acc) = acc + 1

function cell_position(outside, categories, lookups, d, i, stride, acc)
    x = first(d)
    k = get(first(lookups), lookup_key(x), 0)
    k == 0 && (k = unknown_category(outside, i, x, first(categories)))
    k == 0 && return 0
    return cell_position(outside, Base.tail(categories), Base.tail(lookups), Base.tail(d),
                         i + 1, stride * length(first(categories)), acc + (k - 1) * stride)
end

"""
    cell_index(archive::CategoricalArchive, d)

Linear index of the cell whose categories equal the components of descriptor
`d`, a tuple or a vector. Throws an `ArgumentError` if `d` has the wrong number
of dimensions. What happens to a component that is not a category depends on the
archive's `outside` policy: `ThrowOutside()` throws and `RejectOutside()`
returns `0`, which is not a cell.
"""
function cell_index(archive::CategoricalArchive{G,F,D}, d::Tuple) where {G,F,D}
    length(d) == fieldcount(D) ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $(fieldcount(D))"))
    return cell_position(archive.outside, archive.categories, archive.lookups, d, 1, 1, 0)
end

function cell_index(archive::CategoricalArchive{G,F,D}, d::AbstractVector) where {G,F,D}
    length(d) == fieldcount(D) ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $(fieldcount(D))"))
    return cell_index(archive, ntuple(i -> d[i], Val(fieldcount(D))))
end

stored_descriptor(::CategoricalArchive{G,F,D}, d::Tuple) where {G,F,D} = convert(D, d)

function stored_descriptor(archive::CategoricalArchive{G,F,D}, d::AbstractVector) where {G,F,D}
    return stored_descriptor(archive, ntuple(i -> d[i], Val(fieldcount(D))))
end
