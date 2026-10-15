"""
    Evaluation(fitness, descriptor)
    Evaluation(fitness, descriptor, phenotype)
    Evaluation{F,D,P}(fitness, descriptor, phenotype)

The result of evaluating one solution.

`fitness` is a `Real` for a single objective or an `AbstractVector` for
several. Fitness is maximized; a problem that minimizes returns the negated
value. `descriptor` locates the solution in descriptor space and is usually an
`SVector{N,Float64}`.

`phenotype` is what the problem computed from the genotype and described, when
the two differ: the product of a reaction sequence, the shape that a set of
assembly rules builds. An archive whose phenotype type is `P` stores it next to
the genotype, and [`elites`](@ref QualityDiversity.elites) returns it, so it
need not be computed again. Variation sees only genotypes. Without a phenotype
the field is `nothing` and `P` is `Nothing`, which takes no memory in an
archive.

# Example

```jldoctest
julia> using QualityDiversity, StaticArrays

julia> Evaluation(1.0, SVector(0.5, 0.5)).phenotype === nothing
true

julia> Evaluation(1.0, SVector(0.5, 0.5), "CCO").phenotype
"CCO"
```
"""
struct Evaluation{F,D,P}
    fitness::F
    descriptor::D
    phenotype::P
end

Evaluation(fitness, descriptor) = Evaluation(fitness, descriptor, nothing)

Evaluation{F,D}(fitness, descriptor) where {F,D} = Evaluation{F,D,Nothing}(fitness, descriptor, nothing)

"""
    AddStatus

Outcome of inserting a solution into an archive.

- `Rejected`: the archive left unchanged.
- `Improved`: the solution replaced a worse occupant of its cell.
- `NewCell`: the solution filled a cell that was empty.
"""
@enum AddStatus Rejected Improved NewCell

"""
    AddResult{F}(status, improvement)

Result of [`add!`](@ref QualityDiversity.add!).

`improvement` is the fitness gain over the previous occupant of the cell, or
over the archive's threshold for a cell that was empty. It is zero or
negative for a rejected solution. Emitters that rank
candidates by improvement, such as CMA-ME, read this field.
"""
struct AddResult{F}
    status::AddStatus
    improvement::F
end

# The phenotype of an evaluation as an archive with phenotype type `P` stores it. A
# mismatch throws before the archive changes.
stored_phenotype(::Type{P}, phenotype) where {P} = convert(P, phenotype)

function stored_phenotype(::Type{P}, ::Nothing) where {P}
    Nothing <: P && return nothing
    throw(ArgumentError("the evaluation has no phenotype but the archive stores phenotypes of type $P; return one from the problem, or build the archive without phenotype"))
end

function stored_phenotype(::Type{Nothing}, phenotype)
    throw(ArgumentError("the evaluation has a phenotype of type $(typeof(phenotype)) but the archive stores none; build the archive with phenotype=$(typeof(phenotype))"))
end

stored_phenotype(::Type{Nothing}, ::Nothing) = nothing

# Writes a phenotype into the storage of an archive. A `Vector{Nothing}` holds only
# `nothing`, so the write is skipped: an archive without phenotypes then compiles to the
# code it would have without the field, bounds check included.
store_phenotype!(phenotypes::AbstractVector, i, phenotype) = (phenotypes[i] = phenotype; nothing)
store_phenotype!(::Vector{Nothing}, i, ::Nothing) = nothing

# Moves the phenotype in slot `from` to slot `to`.
move_phenotype!(phenotypes::AbstractVector, to, from) = (phenotypes[to] = phenotypes[from]; nothing)
move_phenotype!(::Vector{Nothing}, to, from) = nothing
