"""
    GradientOptimizer

Supertype of the rules by which a [`GradientArborescenceEmitter`](@ref) moves its solution
point along the step that a round of candidates found: [`AdamAscent`](@ref) and
[`GradientAscent`](@ref). They ascend, since fitness is maximized.
"""
abstract type GradientOptimizer end

"""
    GradientAscent(learning_rate)

Move the solution point `θ` by the step `Δ` of a round, scaled: `θ ← θ + learning_rate * Δ`.
This is the update of CMA-MEGA in Algorithm 1 of Fontaine and Nikolaidis (2021), who use
`learning_rate = 1`. The size of a move follows the size of `Δ`, so the rate depends on
the problem; [`AdamAscent`](@ref) depends on it less.
"""
struct GradientAscent <: GradientOptimizer
    learning_rate::Float64
    function GradientAscent(learning_rate::Real)
        learning_rate > 0 && isfinite(learning_rate) ||
            throw(ArgumentError("learning_rate must be positive and finite, got $learning_rate"))
        return new(learning_rate)
    end
end

"""
    AdamAscent(; learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8)

Move the solution point with Adam (Kingma and Ba, 2015), as CMA-MEGA (Adam) of Fontaine
and Nikolaidis (2021) does: Adam keeps decaying averages of the steps (rate `beta1`) and
of their squares (rate `beta2`), and moves every entry of the solution by about
`learning_rate` in the direction of the average step, less where the steps disagree.
`learning_rate = 0.002` is the rate of every experiment of the paper. The update is that of
pyribs, with `epsilon` added to the square root of the second average.
"""
struct AdamAscent <: GradientOptimizer
    learning_rate::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    function AdamAscent(learning_rate::Real, beta1::Real, beta2::Real, epsilon::Real)
        learning_rate > 0 && isfinite(learning_rate) ||
            throw(ArgumentError("learning_rate must be positive and finite, got $learning_rate"))
        0 <= beta1 < 1 || throw(ArgumentError("beta1 must be in [0, 1), got $beta1"))
        0 <= beta2 < 1 || throw(ArgumentError("beta2 must be in [0, 1), got $beta2"))
        epsilon > 0 && isfinite(epsilon) ||
            throw(ArgumentError("epsilon must be positive and finite, got $epsilon"))
        return new(learning_rate, beta1, beta2, epsilon)
    end
end

AdamAscent(; learning_rate::Real=0.002, beta1::Real=0.9, beta2::Real=0.999, epsilon::Real=1e-8) =
    AdamAscent(learning_rate, beta1, beta2, epsilon)

# The solution point and the moment estimates of Adam, which `GradientAscent` leaves at zero.
mutable struct AscentState
    const theta::Vector{Float64}
    const m::Vector{Float64}
    const v::Vector{Float64}
    t::Int
end

AscentState(dim::Integer) = AscentState(zeros(dim), zeros(dim), zeros(dim), 0)

function reset_ascent!(state::AscentState, theta0)
    copyto!(state.theta, theta0)
    fill!(state.m, 0)
    fill!(state.v, 0)
    state.t = 0
    return state
end

function ascend!(state::AscentState, opt::GradientAscent, step)
    state.theta .+= opt.learning_rate .* step
    return state
end

function ascend!(state::AscentState, opt::AdamAscent, step)
    state.t += 1
    (; beta1, beta2) = opt
    rate = opt.learning_rate * sqrt(1 - beta2^state.t) / (1 - beta1^state.t)
    @. state.m = beta1 * state.m + (1 - beta1) * step
    @. state.v = beta2 * state.v + (1 - beta2) * step^2
    @. state.theta += rate * state.m / (sqrt(state.v) + opt.epsilon)
    return state
end
