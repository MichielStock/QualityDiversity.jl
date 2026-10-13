"""
    GaussianMutation(σ)

Mutation that adds independent `Normal(0, σ)` noise to every entry of a
real-valued vector. `σ` must be positive.

`vary(rng, op, parent)` returns a new `Vector` of floating-point numbers and
leaves `parent` unchanged.
"""
struct GaussianMutation{T<:Real} <: AbstractVariation
    σ::T
    function GaussianMutation{T}(σ) where {T<:Real}
        σ > 0 || throw(ArgumentError("σ must be positive, got $σ"))
        return new{T}(σ)
    end
end

GaussianMutation(σ::Real) = GaussianMutation{typeof(σ)}(σ)

function vary(rng::AbstractRNG, op::GaussianMutation, parent::AbstractVector{<:Real})
    T = float(promote_type(eltype(parent), typeof(op.σ)))
    child = Vector{T}(undef, length(parent))
    for i in eachindex(parent)
        child[i] = parent[i] + op.σ * randn(rng, T)
    end
    return child
end
