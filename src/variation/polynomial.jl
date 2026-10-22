"""
    PolynomialMutation(lower, upper; eta=20.0, proportion=1.0)

Mutation for real-valued vectors in a box, the bounded polynomial mutation of
Deb and Goyal (1996) used in NSGA-II. Each entry is changed with probability
`proportion`. A changed entry moves by a random step whose density falls off
polynomially with the distance, and the step is drawn so that the result never
leaves `[lower, upper]`: near a bound the steps shrink on that side, and an
entry exactly on a bound only moves inward. Larger `eta` gives smaller steps; the
usual values are 5 to 50. A step has size about `(upper - lower) / (eta + 1)`.

`lower` and `upper` are numbers, the same for every entry, or vectors with one
bound per entry. Both must be finite and `lower < upper`. `eta` must be
non-negative and `proportion` between 0 and 1; the textbook value of
`proportion` is `1 / n` for vectors of length `n`.

`vary(rng, op, parent)` requires every entry of `parent` to lie inside the
bounds, and throws an `ArgumentError` otherwise; an archive fed by other operators or by
a `random_solution` that ignore the bounds can hold such a parent, so give every
emitter of a run the same bounds. It returns a new array with the
indices of `parent`, made with `similar` (a `Vector` for a `Vector`), with the
element type of `parent` (`Float64` for integers), and leaves `parent` unchanged.

Changing an entry takes two floating-point powers, about 50 ns, so a child of 20
entries costs around 1 µs where a `GaussianMutation` takes under 100 ns; a
`proportion` below 1 reduces it in proportion.

The mutation in QDax (`polynomial_mutation`) differs from the original in the
power term, `δ` where the original has `1 - δ`. There, an entry on a bound never
moves, and one near it is often clipped onto it. This implementation follows the
original, as pymoo does.

# Example

```jldoctest
julia> using QualityDiversity, Random

julia> op = PolynomialMutation(-5.12, 5.12; eta=20);

julia> child = QualityDiversity.vary(Xoshiro(1), op, [5.12, 0.0, -5.12]);

julia> all(-5.12 .<= child .<= 5.12)
true
```
"""
struct PolynomialMutation{L,U} <: AbstractVariation
    lower::L
    upper::U
    eta::Float64
    proportion::Float64
    function PolynomialMutation{L,U}(lower, upper, eta, proportion) where {L,U}
        all(isfinite, lower) && all(isfinite, upper) && all(lower .< upper) ||
            throw(ArgumentError("lower must be finite and below upper, got lower = $lower and upper = $upper"))
        eta >= 0 || throw(ArgumentError("eta must be non-negative, got $eta"))
        0 <= proportion <= 1 ||
            throw(ArgumentError("proportion must be between 0 and 1, got $proportion"))
        return new{L,U}(lower, upper, eta, proportion)
    end
end

function PolynomialMutation(lower::Union{Real,AbstractVector{<:Real}},
                            upper::Union{Real,AbstractVector{<:Real}};
                            eta::Real=20.0, proportion::Real=1.0)
    length(lower) == length(upper) || lower isa Real || upper isa Real ||
        throw(ArgumentError("lower has $(length(lower)) entries but upper has $(length(upper))"))
    return PolynomialMutation{typeof(lower),typeof(upper)}(lower, upper, Float64(eta), Float64(proportion))
end

# The bound of the `k`-th entry of a solution, counted from one.
bound(b::Real, k) = b
bound(b::AbstractVector, k) = b[firstindex(b)+k-1]

# The step of Deb's bounded polynomial mutation for an entry `x` in `[lo, hi]`,
# given a uniform number `u`. `x + step` stays inside `[lo, hi]` up to rounding.
function polynomial_step(x::T, lo::T, hi::T, eta::T, u::T) where {T<:AbstractFloat}
    span = hi - lo
    power = inv(eta + 1)
    if u < T(0.5)
        value = 2u + (1 - 2u) * (1 - (x - lo) / span)^(eta + 1)
        return (value^power - 1) * span
    else
        value = 2 * (1 - u) + 2 * (u - T(0.5)) * (1 - (hi - x) / span)^(eta + 1)
        return (1 - value^power) * span
    end
end

function vary(rng::AbstractRNG, op::PolynomialMutation, parent::AbstractVector{<:Real})
    n = length(parent)
    for (name, b) in (("lower", op.lower), ("upper", op.upper))
        b isa AbstractVector && length(b) != n &&
            throw(ArgumentError("$name has $(length(b)) entries but the parent has $n"))
    end
    T = float(eltype(parent))
    eta = T(op.eta)
    child = similar(parent, T)
    for (k, i) in enumerate(eachindex(child, parent))
        x = T(parent[i])
        lo = T(bound(op.lower, k))
        hi = T(bound(op.upper, k))
        lo <= x <= hi ||
            throw(ArgumentError("parent entry $i is $x, outside the bounds [$lo, $hi]"))
        if rand(rng) < op.proportion
            child[i] = clamp(x + polynomial_step(x, lo, hi, eta, rand(rng, T)), lo, hi)
        else
            child[i] = x
        end
    end
    return child
end
