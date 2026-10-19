"""
    GradientOperatorEmitter(; sigma, batch_size=64, normalize=true)

Emitter of OMG-MEGA (Fontaine and Nikolaidis, 2021, "Differentiable Quality Diversity"),
MAP-Elites with a gradient step in place of a random mutation. It works on a
[`GradientArchive`](@ref), which keeps the gradients of every elite. For each candidate it
draws an elite `θ` uniformly over the occupied cells and coefficients `c` from a normal
distribution with mean zero and standard deviation `sigma`, takes the absolute value of
`c₀` so that the step never lowers the fitness to first order, and returns
`θ + c₀∇f + Σⱼ cⱼ∇mⱼ`, with the gradients of the fitness, `∇f`, and of the descriptor
components, `∇mⱼ`, at `θ`, normalized to unit length unless `normalize=false`. The paper
finds the normalization essential and uses `sigma` from `1` to `10`.

`sigma` is the standard deviation of the coefficients, as in pyribs; QDax takes its
square. The solutions must be real vectors.
"""
struct GradientOperatorEmitter <: AbstractEmitter
    sigma::Float64
    batch_size::Int
    normalize::Bool
    function GradientOperatorEmitter(sigma::Real, batch_size::Integer, normalize::Bool)
        sigma > 0 && isfinite(sigma) || throw(ArgumentError("sigma must be positive and finite, got $sigma"))
        batch_size >= 1 || throw(ArgumentError("batch_size must be at least 1, got $batch_size"))
        return new(sigma, batch_size, normalize)
    end
end

GradientOperatorEmitter(; sigma::Real, batch_size::Integer=64, normalize::Bool=true) =
    GradientOperatorEmitter(sigma, batch_size, normalize)

batch_size(emitter::GradientOperatorEmitter) = emitter.batch_size

"""
    ask(rng, emitter::GradientOperatorEmitter, archive::GradientArchive) -> Vector{G}

`batch_size` children, each a gradient step from an elite drawn uniformly from `archive`.
"""
function ask(rng::AbstractRNG, emitter::GradientOperatorEmitter, archive::GradientArchive{G}) where {G}
    cells = gradient_storage(archive.inner)
    isempty(cells.occupied_idx) && throw(ArgumentError("cannot sample elites from an empty archive"))
    xs = Vector{G}(undef, emitter.batch_size)
    for k in eachindex(xs)
        i = rand(rng, cells.occupied_idx)
        jacobian = archive.gradients[i]
        child = collect(Float64, cells.genotypes[i])
        for r in axes(jacobian, 1)
            c = emitter.sigma * randn(rng)
            r == firstindex(jacobian, 1) && (c = abs(c))
            gradient = view(jacobian, r, :)
            emitter.normalize && (c /= norm(gradient) + GRADIENT_EPSILON)
            child .+= c .* gradient
        end
        xs[k] = convert(G, child)
    end
    return xs
end

function ask(rng::AbstractRNG, emitter::GradientOperatorEmitter, archive::AbstractArchive)
    throw(ArgumentError("a GradientOperatorEmitter steps along the gradients of the elites, which only a GradientArchive keeps, not a $(nameof(typeof(archive)))"))
end
