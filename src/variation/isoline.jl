"""
    IsoLineVariation(σ_iso, σ_line)
    IsoLineVariation(; σ_iso=0.01, σ_line=0.2)

Crossover for real-valued vectors that moves along the line through two parents
(Vassiliades and Mouret, 2018). For parents `p1` and `p2` the child is

    p1 + σ_iso * z + σ_line * u * (p2 - p1)

with `z` a vector of independent standard normal numbers and `u` one standard
normal number. The first noise term spreads the child around `p1` in every
direction (isotropic), the second stretches it along the line to `p2`. Both
standard deviations must be non-negative and finite; `σ_line = 0` leaves a Gaussian
mutation of `p1`, and both zero a copy of `p1`.

The two parents must have the same indices. `vary(rng, op, p1, p2)` returns a
new array with those indices, made with `similar` from `p1` (a `Vector` for a
`Vector`), whose entries have the floating-point type that the entries of the two
parents promote to: `Float32` for two `Vector{Float32}`s. The standard deviations
are converted to that type. The parents are left unchanged.

# Example

```julia
prob = QualityDiversity.Benchmarks.Rastrigin(10)
alg = MAPElites(prob; dims=(20, 20), variation=IsoLineVariation(0.01, 0.2))
```
"""
struct IsoLineVariation{T<:Real} <: AbstractVariation
    σ_iso::T
    σ_line::T
    function IsoLineVariation{T}(σ_iso, σ_line) where {T<:Real}
        σ_iso >= 0 && isfinite(σ_iso) ||
            throw(ArgumentError("σ_iso must be non-negative and finite, got $σ_iso"))
        σ_line >= 0 && isfinite(σ_line) ||
            throw(ArgumentError("σ_line must be non-negative and finite, got $σ_line"))
        return new{T}(σ_iso, σ_line)
    end
end

IsoLineVariation(σ_iso::Real, σ_line::Real) = IsoLineVariation{promote_type(typeof(σ_iso), typeof(σ_line))}(σ_iso, σ_line)

IsoLineVariation(; σ_iso::Real=0.01, σ_line::Real=0.2) = IsoLineVariation(σ_iso, σ_line)

n_parents(::IsoLineVariation) = 2

function check_parent_axes(p1, p2)
    length(p1) == length(p2) ||
        throw(ArgumentError("parents have $(length(p1)) and $(length(p2)) entries but must have the same number"))
    axes(p1) == axes(p2) ||
        throw(ArgumentError("parents have the indices $(firstindex(p1)):$(lastindex(p1)) and $(firstindex(p2)):$(lastindex(p2)) but must have the same"))
    return nothing
end

function vary(rng::AbstractRNG, op::IsoLineVariation,
              p1::AbstractVector{<:Real}, p2::AbstractVector{<:Real})
    check_parent_axes(p1, p2)
    # The child keeps the float type of the parents, as `GaussianMutation` does.
    T = float(promote_type(eltype(p1), eltype(p2)))
    σ_iso = convert(T, op.σ_iso)
    line = convert(T, op.σ_line) * randn(rng, T)
    child = similar(p1, T)
    for i in eachindex(child, p1, p2)
        child[i] = p1[i] + σ_iso * randn(rng, T) + line * (p2[i] - p1[i])
    end
    return child
end
