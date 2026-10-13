"""
    UniformCrossover(; proportion=0.5)

Crossover for genotypes that are vectors of any element type. Each entry of the
child is taken from the second parent with probability `proportion`, and from
the first parent otherwise. With `proportion = 0` the child copies the first
parent and with `1` the second. The parents must have the same length.

It works for real numbers, integers, symbols or any other entries, since it
only copies them. It does not create values that neither parent has; combine it
with a mutation for that, for example in two emitters.

`vary(rng, op, p1, p2)` returns a new `Vector` whose element type holds the
entries of both parents, and leaves the parents unchanged.

QDax calls its version of this operator `polynomial_crossover`; it copies a
random subset of entries, drawn with replacement, from the second parent.

# Example

```jldoctest
julia> using QualityDiversity, Random

julia> child = QualityDiversity.vary(Xoshiro(1), UniformCrossover(), [1, 2, 3, 4], [5, 6, 7, 8]);

julia> all(c -> c in (1, 5) || c in (2, 6) || c in (3, 7) || c in (4, 8), child)
true
```
"""
struct UniformCrossover <: AbstractVariation
    proportion::Float64
    function UniformCrossover(proportion::Real)
        0 <= proportion <= 1 ||
            throw(ArgumentError("proportion must be between 0 and 1, got $proportion"))
        return new(Float64(proportion))
    end
end

UniformCrossover(; proportion::Real=0.5) = UniformCrossover(proportion)

n_parents(::UniformCrossover) = 2

function vary(rng::AbstractRNG, op::UniformCrossover, p1::AbstractVector, p2::AbstractVector)
    length(p1) == length(p2) ||
        throw(ArgumentError("parents have $(length(p1)) and $(length(p2)) entries but must have the same number"))
    child = Vector{promote_type(eltype(p1), eltype(p2))}(undef, length(p1))
    for i in eachindex(p1, p2)
        child[i] = rand(rng) < op.proportion ? p2[i] : p1[i]
    end
    return child
end
