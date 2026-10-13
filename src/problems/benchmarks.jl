const RASTRIGIN_LIMIT = 5.12

"""
    Rastrigin(dim)

The Rastrigin function as a quality-diversity benchmark over `dim` real
parameters, `dim ≥ 2`.

The solution is clipped to `[-5.12, 5.12]` before it is scored. Fitness is the
negated Rastrigin value, so the maximum is `0` at the origin. The descriptor is
the mean of the first `dim ÷ 2` clipped parameters and the mean of the
remaining ones, which both lie in `[-5.12, 5.12]`.

Solutions are `Vector{Float64}`, drawn uniformly from the clipping box.
"""
struct Rastrigin <: AbstractQDProblem
    dim::Int
    function Rastrigin(dim::Integer)
        dim >= 2 ||
            throw(ArgumentError("Rastrigin needs at least 2 dimensions for its 2-D descriptor, got $dim"))
        return new(dim)
    end
end

function evaluate(prob::Rastrigin, x::AbstractVector{<:Real})
    n = prob.dim
    length(x) == n ||
        throw(ArgumentError("solution has $(length(x)) entries but Rastrigin has dimension $n"))
    h = n ÷ 2
    total = 0.0
    first_half = 0.0
    second_half = 0.0
    for i in 1:n
        xi = clamp(Float64(x[i]), -RASTRIGIN_LIMIT, RASTRIGIN_LIMIT)
        total += xi^2 - 10 * cos(2π * xi)
        if i <= h
            first_half += xi
        else
            second_half += xi
        end
    end
    descriptor = SVector(clamp(first_half / h, -RASTRIGIN_LIMIT, RASTRIGIN_LIMIT),
                         clamp(second_half / (n - h), -RASTRIGIN_LIMIT, RASTRIGIN_LIMIT))
    return Evaluation(-(10 * n + total), descriptor)
end

function random_solution(rng::AbstractRNG, prob::Rastrigin)
    return (2 * rand(rng, prob.dim) .- 1) .* RASTRIGIN_LIMIT
end

function descriptor_bounds(::Rastrigin)
    return ((-RASTRIGIN_LIMIT, RASTRIGIN_LIMIT), (-RASTRIGIN_LIMIT, RASTRIGIN_LIMIT))
end

# Each clipped term of the Rastrigin sum is at most 40.5.
fitness_offset(prob::Rastrigin) = 41.0 * prob.dim
