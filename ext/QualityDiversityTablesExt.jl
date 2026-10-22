"""
    QualityDiversityTablesExt

Tables.jl support for archives. With Tables loaded, every
[`AbstractArchive`](@ref QualityDiversity.AbstractArchive) is a table with one row per
elite, in the order of [`elites`](@ref QualityDiversity.elites), so `DataFrame(archive)`
and `Tables.rowtable(archive)` work. The columns are

- `genotype`: the stored genotype, not a copy;
- `fitness`: the fitness, a single column also for a vector of objectives;
- `descriptor_1`, ..., `descriptor_N`: one column per descriptor dimension, for an
  `SVector` or a tuple descriptor;
- `phenotype`: the stored phenotype, only for an archive that stores one.

The column types follow from the type parameters of the archive, so an empty archive
has the same columns as a full one. An archive whose descriptor type has no length in
its type, such as a `Vector`, throws an `ArgumentError`.

# Example

```jldoctest
julia> using QualityDiversity, StaticArrays, Tables

julia> archive = GridArchive((2, 2), ((0, 1), (0, 1)));

julia> QualityDiversity.add!(archive, [0.5], Evaluation(1.0, SVector(0.25, 0.75)));

julia> Tables.columntable(archive)
(genotype = [[0.5]], fitness = [1.0], descriptor_1 = [0.25], descriptor_2 = [0.75])
```
"""
module QualityDiversityTablesExt

using QualityDiversity: QualityDiversity, AbstractArchive, elites
using StaticArrays: StaticVector
using Tables: Tables

Tables.istable(::Type{<:AbstractArchive}) = true
Tables.rowaccess(::Type{<:AbstractArchive}) = true
Tables.schema(archive::AbstractArchive) = Tables.schema(Tables.rows(archive))

# The rows of an archive: one `NamedTuple` per elite, with the column names and types in
# the type, so that building the columns of an empty archive needs no row to look at.
struct ArchiveRows{names,types,P,E}
    elites::E
end

function Tables.rows(archive::AbstractArchive{G,F,D,P}) where {G,F,D,P}
    dtypes = descriptor_types(D)
    N = length(dtypes)
    names = (:genotype, :fitness, ntuple(i -> Symbol(:descriptor_, i), N)..., phenotype_name(P)...)
    types = Tuple{G,F,dtypes...,phenotype_type(P)...}
    iterator = elites(archive)
    return ArchiveRows{names,types,P,typeof(iterator)}(iterator)
end

# The element type of each descriptor column, read from the descriptor type alone.
descriptor_types(::Type{<:StaticVector{N,T}}) where {N,T} = ntuple(_ -> T, Val(N))
descriptor_types(::Type{D}) where {D<:Tuple} = fieldtypes(D)
function descriptor_types(::Type{D}) where {D}
    throw(ArgumentError("a table needs one column per descriptor dimension, but the descriptor type $D " *
                        "does not say how many there are; use an SVector or a tuple as descriptor"))
end

phenotype_name(::Type{Nothing}) = ()
phenotype_name(::Type) = (:phenotype,)
phenotype_type(::Type{Nothing}) = ()
phenotype_type(::Type{P}) where {P} = (P,)

row_values(x, ev, ::Type{Nothing}) = (x, ev.fitness, Tuple(ev.descriptor)...)
row_values(x, ev, ::Type) = (x, ev.fitness, Tuple(ev.descriptor)..., ev.phenotype)

Tables.istable(::Type{<:ArchiveRows}) = true
Tables.rowaccess(::Type{<:ArchiveRows}) = true
Tables.rows(rows::ArchiveRows) = rows
Tables.schema(::ArchiveRows{names,types}) where {names,types} = Tables.Schema{names,types}()

Base.eltype(::Type{<:ArchiveRows{names,types}}) where {names,types} = NamedTuple{names,types}
# The rows have a length when the elites do; `elites` of a `ParetoGridArchive` is a
# flattened iterator, whose length is unknown.
Base.IteratorSize(::Type{<:ArchiveRows{names,types,P,E}}) where {names,types,P,E} = rows_size(Base.IteratorSize(E))
rows_size(::Base.HasShape) = Base.HasLength()
rows_size(size::Base.IteratorSize) = size
Base.length(rows::ArchiveRows) = length(rows.elites)

function Base.iterate(rows::ArchiveRows{names,types,P}, state...) where {names,types,P}
    next = iterate(rows.elites, state...)
    next === nothing && return nothing
    (x, ev), s = next
    return NamedTuple{names}(row_values(x, ev, P)), s
end

end
