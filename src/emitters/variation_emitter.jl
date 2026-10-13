"""
    VariationEmitter(op; batch_size)

Emitter that draws parents uniformly from the archive and applies the variation
operator `op`, giving `batch_size` children. A mutation takes one parent per
child and a crossover two, as [`n_parents(op)`](@ref QualityDiversity.n_parents)
says; the parents are drawn with replacement. It keeps no state, so it is the
emitter of MAP-Elites.
"""
struct VariationEmitter{V<:AbstractVariation} <: AbstractEmitter
    op::V
    batch_size::Int
    function VariationEmitter(op::V, batch_size::Integer) where {V<:AbstractVariation}
        batch_size >= 1 ||
            throw(ArgumentError("batch_size must be at least 1, got $batch_size"))
        return new{V}(op, batch_size)
    end
end

VariationEmitter(op::AbstractVariation; batch_size::Integer) = VariationEmitter(op, batch_size)

batch_size(emitter::VariationEmitter) = emitter.batch_size

function ask(rng::AbstractRNG, emitter::VariationEmitter, archive::AbstractArchive{G}) where {G}
    children = Vector{G}(undef, emitter.batch_size)
    return vary_batch!(children, rng, emitter.op, archive, Val(n_parents(emitter.op)))
end

function vary_batch!(children, rng, op, archive, ::Val{1})
    parents = sample_elites(rng, archive, length(children))
    for i in eachindex(children, parents)
        children[i] = vary(rng, op, parents[i])
    end
    return children
end

function vary_batch!(children, rng, op, archive, ::Val{2})
    first_parents = sample_elites(rng, archive, length(children))
    second_parents = sample_elites(rng, archive, length(children))
    for i in eachindex(children, first_parents, second_parents)
        children[i] = vary(rng, op, first_parents[i], second_parents[i])
    end
    return children
end
