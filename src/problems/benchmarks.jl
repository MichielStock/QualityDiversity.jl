"""
    QualityDiversity.Benchmarks

The benchmark problems of the package: [`Rastrigin`](@ref), [`Sphere`](@ref) and
[`BiSphere`](@ref). They are not exported from `QualityDiversity`, because `Sphere` is
also a name of Makie and GeometryBasics; load them with

```julia
using QualityDiversity, QualityDiversity.Benchmarks
prob = Rastrigin(10)
```
"""
module Benchmarks

using ..QualityDiversity: QualityDiversity, AbstractQDProblem, Evaluation
using Random: AbstractRNG
using StaticArrays: SVector

export Rastrigin, Sphere, BiSphere

const RASTRIGIN_LIMIT = 5.12

# The number type `evaluate` computes in: `Float64`, or the element type of the solution
# when that carries more, such as the dual numbers of ForwardDiff or `BigFloat`.
compute_type(x::AbstractVector) = promote_type(Float64, eltype(x))

# The number type a device kernel computes in: that of the solution, so that `Float32`
# solutions are scored in `Float32` on a GPU. The fitness can then differ slightly from
# that of `evaluate`.
device_compute_type(x::AbstractVector) = float(eltype(x))

"""
    Rastrigin(dim)

The Rastrigin function as a quality-diversity benchmark over `dim` real
parameters, `dim ≥ 2`.

The solution is clipped to `[-5.12, 5.12]` before it is scored. Fitness is the
negated Rastrigin value, so the maximum is `0` at the origin. The descriptor is
the mean of the first `dim ÷ 2` clipped parameters and the mean of the
remaining ones, which both lie in `[-5.12, 5.12]`.

Solutions are `Vector{Float64}`, drawn uniformly from the clipping box.

`evaluate` computes in `Float64`, or in the element type of the solution when that
carries more. On a device batch the kernel computes in the element type of the
solution, so `Float32` solutions are scored in `Float32` there, and their fitness can
differ from that of `evaluate` in the last digits.
"""
struct Rastrigin <: AbstractQDProblem
    dim::Int
    function Rastrigin(dim::Integer)
        dim >= 2 ||
            throw(ArgumentError("Rastrigin needs at least 2 dimensions for its 2-D descriptor, got $dim"))
        return new(dim)
    end
end

function QualityDiversity.evaluate(prob::Rastrigin, x::AbstractVector{<:Real})
    length(x) == prob.dim ||
        throw(ArgumentError("solution has $(length(x)) entries but Rastrigin has dimension $(prob.dim)"))
    return rastrigin(prob, x, compute_type(x))
end

# Without the length check, whose message a device kernel cannot build; the archive
# checks the length of the solutions of a device batch.
function QualityDiversity.device_evaluate(prob::Rastrigin, x::AbstractVector{<:Real})
    return rastrigin(prob, x, device_compute_type(x))
end

# The evaluation computed in `T`: every constant is converted to `T` first, since one
# `Float64` constant would promote a `Float32` computation.
function rastrigin(prob::Rastrigin, x::AbstractVector{<:Real}, ::Type{T}) where {T}
    n = prob.dim
    h = n ÷ 2
    limit = convert(T, RASTRIGIN_LIMIT)
    two_pi = convert(T, 2π)
    total = zero(T)
    first_half = zero(T)
    second_half = zero(T)
    for i in 1:n
        xi = clamp(convert(T, x[i]), -limit, limit)
        total += xi^2 - 10 * cos(two_pi * xi)
        if i <= h
            first_half += xi
        else
            second_half += xi
        end
    end
    descriptor = SVector(clamp(first_half / h, -limit, limit), clamp(second_half / (n - h), -limit, limit))
    return Evaluation(-(10 * n + total), descriptor)
end

function QualityDiversity.random_solution(rng::AbstractRNG, prob::Rastrigin)
    return (2 * rand(rng, prob.dim) .- 1) .* RASTRIGIN_LIMIT
end

function QualityDiversity.descriptor_bounds(::Rastrigin)
    return ((-RASTRIGIN_LIMIT, RASTRIGIN_LIMIT), (-RASTRIGIN_LIMIT, RASTRIGIN_LIMIT))
end

# Each clipped term of the Rastrigin sum is at most 40.5.
QualityDiversity.fitness_offset(prob::Rastrigin) = 41.0 * prob.dim

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

`evaluate` computes in `Float64`, or in the element type of the solution when that
carries more. On a device batch the kernel computes in the element type of the
solution, so `Float32` solutions are scored in `Float32` there, and their fitness can
differ from that of `evaluate` in the last digits.
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
function sphere_clip(x)
    limit = convert(typeof(x), SPHERE_LIMIT)
    return -limit <= x <= limit ? x : limit / x
end

function QualityDiversity.evaluate(prob::Sphere, x::AbstractVector{<:Real})
    length(x) == prob.dim ||
        throw(ArgumentError("solution has $(length(x)) entries but Sphere has dimension $(prob.dim)"))
    return sphere(prob, x, compute_type(x))
end

function QualityDiversity.device_evaluate(prob::Sphere, x::AbstractVector{<:Real})
    return sphere(prob, x, device_compute_type(x))
end

function sphere(prob::Sphere, x::AbstractVector{<:Real}, ::Type{T}) where {T}
    n = prob.dim
    h = n ÷ 2
    shift = convert(T, SPHERE_SHIFT)
    distance = zero(T)
    first_half = zero(T)
    second_half = zero(T)
    for i in 1:n
        xi = convert(T, x[i])
        distance += (xi - shift)^2
        i <= h ? (first_half += sphere_clip(xi)) : (second_half += sphere_clip(xi))
    end
    worst = convert(T, (SPHERE_LIMIT + SPHERE_SHIFT)^2 * n)
    return Evaluation(100 * (1 - distance / worst), SVector(first_half, second_half))
end

function QualityDiversity.random_solution(rng::AbstractRNG, prob::Sphere)
    return (2 * rand(rng, prob.dim) .- 1) .* SPHERE_LIMIT
end

function QualityDiversity.descriptor_bounds(prob::Sphere)
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

function QualityDiversity.evaluate(prob::BiSphere, x::AbstractVector{<:Real})
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

function QualityDiversity.random_solution(rng::AbstractRNG, prob::BiSphere)
    return (2 * rand(rng, prob.dim) .- 1) .* BISPHERE_LIMIT
end

function QualityDiversity.descriptor_bounds(::BiSphere)
    return ((-BISPHERE_LIMIT, BISPHERE_LIMIT), (-BISPHERE_LIMIT, BISPHERE_LIMIT))
end

function QualityDiversity.reference_point(prob::BiSphere)
    lowest = -prob.dim * (BISPHERE_LIMIT + BISPHERE_CENTER)^2
    return SVector(lowest, lowest)
end

# The gradients in closed form, several times faster than ForwardDiff and available without
# it. At a clipping limit the derivative is the one from inside; ForwardDiff 1 takes the one
# from outside.
function QualityDiversity.evaluate_gradient(prob::Rastrigin, x::AbstractVector{<:Real})
    ev = QualityDiversity.evaluate(prob, x)
    n = prob.dim
    h = n ÷ 2
    # Every entry is written, so the Jacobian is the only allocation.
    jacobian = Matrix{Float64}(undef, 3, n)
    for (k, i) in enumerate(eachindex(x))
        xi = Float64(x[i])
        inside = -RASTRIGIN_LIMIT <= xi <= RASTRIGIN_LIMIT
        jacobian[1, k] = inside ? -(2 * xi + 20π * sin(2π * xi)) : 0.0
        jacobian[2, k] = inside && k <= h ? 1 / h : 0.0
        jacobian[3, k] = inside && k > h ? 1 / (n - h) : 0.0
    end
    return ev, jacobian
end

function QualityDiversity.evaluate_gradient(prob::Sphere, x::AbstractVector{<:Real})
    ev = QualityDiversity.evaluate(prob, x)
    n = prob.dim
    h = n ÷ 2
    worst = (SPHERE_LIMIT + SPHERE_SHIFT)^2 * n
    jacobian = Matrix{Float64}(undef, 3, n)
    for (k, i) in enumerate(eachindex(x))
        xi = Float64(x[i])
        clip = -SPHERE_LIMIT <= xi <= SPHERE_LIMIT ? 1.0 : -SPHERE_LIMIT / xi^2
        jacobian[1, k] = -200 * (xi - SPHERE_SHIFT) / worst
        jacobian[2, k] = k <= h ? clip : 0.0
        jacobian[3, k] = k <= h ? 0.0 : clip
    end
    return ev, jacobian
end

end # module Benchmarks
