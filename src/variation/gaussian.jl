"""
    GaussianMutation(σ)

Mutation that adds independent `Normal(0, σ)` noise to every entry of a
real-valued vector. `σ` must be positive and finite.

`vary(rng, op, parent)` returns a new array of floating-point numbers with the
indices of `parent`, made with `similar` (a `Vector` for a `Vector`), and leaves
`parent` unchanged.
"""
struct GaussianMutation{T<:Real} <: AbstractVariation
    σ::T
    function GaussianMutation{T}(σ) where {T<:Real}
        σ > 0 && isfinite(σ) || throw(ArgumentError("σ must be positive and finite, got $σ"))
        return new{T}(σ)
    end
end

GaussianMutation(σ::Real) = GaussianMutation{typeof(σ)}(σ)

function vary(rng::AbstractRNG, op::GaussianMutation, parent::AbstractVector{<:Real})
    T = float(promote_type(eltype(parent), typeof(op.σ)))
    child = similar(parent, T)
    for i in eachindex(child, parent)
        child[i] = parent[i] + op.σ * randn(rng, T)
    end
    return child
end
