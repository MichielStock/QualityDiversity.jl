"""
    evaluate_gradient(prob::AbstractQDProblem, x) -> (Evaluation, jacobian)

The [`Evaluation`](@ref) of the solution `x` together with the gradients of its fitness and
descriptor, for the algorithms of differentiable quality diversity ([`CMAMEGA`](@ref),
[`CMAMAEGA`](@ref), [`OMGMEGA`](@ref)). Optional: only those algorithms call it.

`x` is a real vector of length `n` and the descriptor has `N` dimensions. `jacobian` is an
`(N + 1) × n` matrix whose first row is the gradient of the fitness and whose row `j + 1` is
the gradient of descriptor component `j`. The fitness is a single real number.

The default differentiates `evaluate(prob, x)` with ForwardDiff.jl, which must be loaded
(`using ForwardDiff`). The code of the problem must then compute in the number type of
`x` rather than convert its entries to `Float64`, since ForwardDiff passes vectors of dual
numbers; a [`QDProblem`](@ref) does, if its function does. A problem defines this method
itself to give its gradients in closed form, which is faster, or to compute them with
another automatic-differentiation package. [`Sphere`](@ref) and [`Rastrigin`](@ref) define
it in closed form. Stochastic problems have no gradient.

# Example

```jldoctest
julia> using QualityDiversity, ForwardDiff

julia> prob = QDProblem(x -> (-sum(abs2, x), (x[1], 2x[2]));
                        init = rng -> randn(rng, 2), bounds = ((-3, 3), (-6, 6)));

julia> ev, jacobian = QualityDiversity.evaluate_gradient(prob, [1.0, 2.0]);

julia> ev.fitness
-5.0

julia> jacobian
3×2 Matrix{Float64}:
 -2.0  -4.0
  1.0   0.0
  0.0   2.0
```
"""
function evaluate_gradient end

evaluate_gradient(prob::AbstractQDProblem, x) = forwarddiff_evaluate_gradient(prob, x)

function evaluate_gradient(prob::AbstractStochasticQDProblem, x)
    throw(ArgumentError("$(nameof(typeof(prob))) is a stochastic problem, which has no gradient; differentiable quality diversity takes deterministic problems"))
end

# `evaluate_gradient` by ForwardDiff, whose extension defines a more specific method.
function forwarddiff_evaluate_gradient(prob, x)
    name = nameof(typeof(prob))
    throw(ArgumentError("$name has no gradient: load ForwardDiff (`using ForwardDiff`) to differentiate it automatically, or define QualityDiversity.evaluate_gradient(prob::$name, x)"))
end

# The fitness followed by the descriptor components of `x`, as one vector: the function
# whose Jacobian `evaluate_gradient` returns. A static vector for a static descriptor, so
# that the passes of ForwardDiff allocate nothing for it.
differentiable_values(prob::AbstractQDProblem, x) = (ev = evaluate(prob, x); stack_values(ev.fitness, ev.descriptor))

stack_values(fitness::Real, descriptor::SVector) = pushfirst(descriptor, fitness)

stack_values(fitness::Real, descriptor) = [fitness; collect(descriptor)]

function stack_values(fitness, descriptor)
    throw(ArgumentError("differentiable quality diversity needs a single real fitness, got a $(typeof(fitness))"))
end

"""
    evaluate_gradient_batch(prob::AbstractQDProblem, xs) -> (evaluations, jacobians)

[`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient) for every solution in `xs`:
a vector of [`Evaluation`](@ref)s and a vector of Jacobian matrices, in the order of `xs`.
Optional: the default calls `evaluate_gradient` on each solution in parallel with
`Threads.@threads`, so it must be thread-safe. A problem specializes it to differentiate a
batch at once. Every solution counts as one evaluation against the budget of a run.
"""
function evaluate_gradient_batch end

function evaluate_gradient_batch(prob::AbstractQDProblem, xs::AbstractVector)
    Base.require_one_based_indexing(xs)
    T = Base.promote_op(evaluate_gradient, typeof(prob), eltype(xs))
    pairs = Vector{T}(undef, length(xs))
    Threads.@threads for i in eachindex(xs)
        pairs[i] = evaluate_gradient(prob, xs[i])
    end
    return map(first, pairs), map(last, pairs)
end

# Checks that the Jacobian of a solution with `n` entries and `N` descriptor dimensions has
# one row per function and one column per entry, and is finite.
function check_jacobian(jacobian, N, n)
    size(jacobian) == (N + 1, n) ||
        throw(ArgumentError("the Jacobian of a solution with $n entries and $N descriptor dimensions must be $(N + 1)×$n, one row for the fitness and one per descriptor dimension, but it is $(join(size(jacobian), "×"))"))
    all(isfinite, jacobian) || throw(ArgumentError("the Jacobian has entries that are not finite"))
    return nothing
end

# Divides every row of `jacobian` by its length, so that each function weighs the same in a
# step; a zero row stays zero.
function normalize_rows!(jacobian::AbstractMatrix)
    for r in axes(jacobian, 1)
        row = view(jacobian, r, :)
        row ./= norm(row) + GRADIENT_EPSILON
    end
    return jacobian
end

# Added to the length of a gradient before dividing by it, as pyribs does.
const GRADIENT_EPSILON = 1e-8
