"""
    GaussianMutation(σ)

Mutation that adds independent `Normal(0, σ)` noise to every entry of a
real-valued vector. `σ` must be positive and finite.

`vary(rng, op, parent)` returns a new array with the indices of `parent`, made with
`similar` (a `Vector` for a `Vector`), whose entries have the floating-point type of
those of `parent`: `Float32` for a `Vector{Float32}`, `Float64` for an integer vector.
`σ` is converted to that type. `parent` is left unchanged.
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
    # The child keeps the float type of the parent: a `Float32` genotype would otherwise
    # be computed in `Float64` and converted back when the archive stores it.
    T = float(eltype(parent))
    σ = convert(T, op.σ)
    child = similar(parent, T)
    for i in eachindex(child, parent)
        child[i] = parent[i] + σ * randn(rng, T)
    end
    return child
end
