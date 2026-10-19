const RASTRIGIN_LIMIT = 5.12

# The number type a benchmark computes in: `Float64`, or the element type of the solution
# when that carries more, such as the dual numbers of ForwardDiff or `BigFloat`.
compute_type(x::AbstractVector) = promote_type(Float64, eltype(x))

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
    length(x) == prob.dim ||
        throw(ArgumentError("solution has $(length(x)) entries but Rastrigin has dimension $(prob.dim)"))
    return device_evaluate(prob, x)
end

# Without the length check, whose message a device kernel cannot build; the archive
# checks the length of the solutions of a device batch.
function device_evaluate(prob::Rastrigin, x::AbstractVector{<:Real})
    n = prob.dim
    h = n ÷ 2
    T = compute_type(x)
    total = zero(T)
    first_half = zero(T)
    second_half = zero(T)
    for i in 1:n
        xi = clamp(convert(T, x[i]), -RASTRIGIN_LIMIT, RASTRIGIN_LIMIT)
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

const SPHERE_LIMIT = 5.12
const SPHERE_SHIFT = 0.4 * SPHERE_LIMIT

"""
    Sphere(dim)

The sphere with a linear projection of Fontaine et al. (2020), a quality-diversity
benchmark over `dim` real parameters, `dim ≥ 2`. Its objective is the negated sphere
function centered at `0.4 * 5.12` in every parameter, scaled so that the optimum has
fitness `100` and the corner `-5.12` of the box has fitness `0`; a solution farther away
has a negative fitness.

The descriptor is the sum of the first `dim ÷ 2` parameters and the sum of the rest,
after each parameter outside `[-5.12, 5.12]` is replaced by `5.12 / θ`. Because
that sum projects many parameters onto one number, most of the solutions land near the
center of the descriptor space, and reaching its edges takes solutions that are
all at the same end of the box. The descriptor lies in `[-5.12 * dim/2, 5.12 * dim/2]`
in both dimensions.

Solutions are `Vector{Float64}`, drawn uniformly from `[-5.12, 5.12]`.
"""
struct Sphere <: AbstractQDProblem
    dim::Int
    function Sphere(dim::Integer)
        dim >= 2 ||
            throw(ArgumentError("Sphere needs at least 2 dimensions for its 2-D descriptor, got $dim"))
        return new(dim)
    end
end

# A parameter inside the box stays; outside it, it is mapped back into the box.
sphere_clip(x) = -SPHERE_LIMIT <= x <= SPHERE_LIMIT ? x : SPHERE_LIMIT / x

function evaluate(prob::Sphere, x::AbstractVector{<:Real})
    length(x) == prob.dim ||
        throw(ArgumentError("solution has $(length(x)) entries but Sphere has dimension $(prob.dim)"))
    return device_evaluate(prob, x)
end

function device_evaluate(prob::Sphere, x::AbstractVector{<:Real})
    n = prob.dim
    h = n ÷ 2
    T = compute_type(x)
    distance = zero(T)
    first_half = zero(T)
    second_half = zero(T)
    for i in 1:n
        xi = convert(T, x[i])
        distance += (xi - SPHERE_SHIFT)^2
        i <= h ? (first_half += sphere_clip(xi)) : (second_half += sphere_clip(xi))
    end
    worst = (SPHERE_LIMIT + SPHERE_SHIFT)^2 * n
    return Evaluation(100 * (1 - distance / worst), SVector(first_half, second_half))
end

function random_solution(rng::AbstractRNG, prob::Sphere)
    return (2 * rand(rng, prob.dim) .- 1) .* SPHERE_LIMIT
end

function descriptor_bounds(prob::Sphere)
    limit = SPHERE_LIMIT * prob.dim / 2
    return ((-limit, limit), (-limit, limit))
end

const BISPHERE_LIMIT = 5.12
const BISPHERE_CENTER = 1.0

"""
    BiSphere(dim)

A benchmark with two objectives over `dim` real parameters, `dim ≥ 2`: the negated squared
distances to the two points `(1, …, 1)` and `(-1, …, -1)`. No solution is best in both; the
solutions that trade them off best lie on the line between the two points.

The solution is clipped to `[-5.12, 5.12]` before it is scored. The fitness is an
`SVector{2,Float64}` of the two objectives, which are at most `0`. The descriptor is that of
[`Rastrigin`](@ref): the mean of the first `dim ÷ 2` clipped parameters and the mean of the
remaining ones. [`reference_point`](@ref QualityDiversity.reference_point) is the lowest value
of both objectives, so every solution adds hypervolume.

Solutions are `Vector{Float64}`, drawn uniformly from the clipping box.
"""
struct BiSphere <: AbstractQDProblem
    dim::Int
    function BiSphere(dim::Integer)
        dim >= 2 ||
            throw(ArgumentError("BiSphere needs at least 2 dimensions for its 2-D descriptor, got $dim"))
        return new(dim)
    end
end

function evaluate(prob::BiSphere, x::AbstractVector{<:Real})
    n = prob.dim
    length(x) == n ||
        throw(ArgumentError("solution has $(length(x)) entries but BiSphere has dimension $n"))
    h = n ÷ 2
    T = compute_type(x)
    near = zero(T)
    far = zero(T)
    first_half = zero(T)
    second_half = zero(T)
    for i in 1:n
        xi = clamp(convert(T, x[i]), -BISPHERE_LIMIT, BISPHERE_LIMIT)
        near += (xi - BISPHERE_CENTER)^2
        far += (xi + BISPHERE_CENTER)^2
        if i <= h
            first_half += xi
        else
            second_half += xi
        end
    end
    descriptor = SVector(clamp(first_half / h, -BISPHERE_LIMIT, BISPHERE_LIMIT),
                         clamp(second_half / (n - h), -BISPHERE_LIMIT, BISPHERE_LIMIT))
    return Evaluation(SVector(-near, -far), descriptor)
end

function random_solution(rng::AbstractRNG, prob::BiSphere)
    return (2 * rand(rng, prob.dim) .- 1) .* BISPHERE_LIMIT
end

function descriptor_bounds(::BiSphere)
    return ((-BISPHERE_LIMIT, BISPHERE_LIMIT), (-BISPHERE_LIMIT, BISPHERE_LIMIT))
end

function reference_point(prob::BiSphere)
    lowest = -prob.dim * (BISPHERE_LIMIT + BISPHERE_CENTER)^2
    return SVector(lowest, lowest)
end
