"""
    IsoLineVariation(σ_iso, σ_line)
    IsoLineVariation(; σ_iso=0.01, σ_line=0.2)

Crossover for real-valued vectors that moves along the line through two parents
(Vassiliades and Mouret, 2018). For parents `p1` and `p2` the child is

    p1 + σ_iso * z + σ_line * u * (p2 - p1)

with `z` a vector of independent standard normal numbers and `u` one standard
normal number. The first noise term spreads the child around `p1` in every
direction (isotropic), the second stretches it along the line to `p2`. Both
standard deviations must be non-negative; `σ_line = 0` leaves a Gaussian
mutation of `p1`, and both zero a copy of `p1`.

The two parents must have the same length. `vary(rng, op, p1, p2)` returns a
new `Vector` of floating-point numbers and leaves the parents unchanged.

# Example

```julia
prob = Rastrigin(10)
alg = MAPElites(prob; dims=(20, 20), variation=IsoLineVariation(0.01, 0.2))
```
"""
struct IsoLineVariation{T<:Real} <: AbstractVariation
    σ_iso::T
    σ_line::T
    function IsoLineVariation{T}(σ_iso, σ_line) where {T<:Real}
        σ_iso >= 0 || throw(ArgumentError("σ_iso must be non-negative, got $σ_iso"))
        σ_line >= 0 || throw(ArgumentError("σ_line must be non-negative, got $σ_line"))
        return new{T}(σ_iso, σ_line)
    end
end

IsoLineVariation(σ_iso::Real, σ_line::Real) = IsoLineVariation{promote_type(typeof(σ_iso), typeof(σ_line))}(σ_iso, σ_line)

IsoLineVariation(; σ_iso::Real=0.01, σ_line::Real=0.2) = IsoLineVariation(σ_iso, σ_line)

n_parents(::IsoLineVariation) = 2

function vary(rng::AbstractRNG, op::IsoLineVariation,
              p1::AbstractVector{<:Real}, p2::AbstractVector{<:Real})
    length(p1) == length(p2) ||
        throw(ArgumentError("parents have $(length(p1)) and $(length(p2)) entries but must have the same number"))
    T = float(promote_type(eltype(p1), eltype(p2), typeof(op.σ_iso)))
    line = op.σ_line * randn(rng, T)
    child = Vector{T}(undef, length(p1))
    for i in eachindex(p1, p2)
        child[i] = p1[i] + op.σ_iso * randn(rng, T) + line * (p2[i] - p1[i])
    end
    return child
end
