"""
    VariationEmitter(op; batch_size)

Emitter that draws `batch_size` parents uniformly from the archive and applies
the mutation `op` to each, giving one child per parent. It keeps no state, so
it is the emitter of MAP-Elites.
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
    parents = sample_elites(rng, archive, emitter.batch_size)
    children = Vector{G}(undef, emitter.batch_size)
    for i in eachindex(parents)
        children[i] = vary(rng, emitter.op, parents[i])
    end
    return children
end
