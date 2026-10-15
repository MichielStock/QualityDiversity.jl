"""
    QDProblem(f; init, bounds)
    QDProblem(f; init, categories)

Wrap a plain function as a problem. `f(x)` returns a `(fitness, descriptor)`
tuple for the solution `x`, or a `(fitness, descriptor, phenotype)` tuple for a
problem whose phenotype an archive should store; see [`Evaluation`](@ref).
`init(rng)` returns one random solution.

With `bounds`, a tuple with one `(lo, hi)` pair per descriptor dimension, the
descriptor is real-valued and `evaluate` returns it as an `SVector{N,Float64}`.
With `categories`, a tuple with one collection of allowed values per dimension,
for example `(0:5, [:alkane, :aromatic, :ring])`, the descriptor is discrete and
`evaluate` returns it as a tuple, which a [`CategoricalArchive`](@ref) takes as
it is.

`f` is called from several threads at once by the default
[`evaluate_batch`](@ref QualityDiversity.evaluate_batch), so it must be
thread-safe.

# Example

```jldoctest
julia> using QualityDiversity, Random

julia> prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                        init = rng -> randn(rng, 2), bounds = ((-3, 3), (-3, 3)));

julia> ev = QualityDiversity.evaluate(prob, [1.0, 2.0]);

julia> ev.fitness, ev.descriptor
(-5.0, [1.0, 2.0])

julia> labelled = QDProblem(x -> (-sum(abs2, x), (length(x), :ring));
                            init = rng -> randn(rng, 2), categories = (1:3, [:chain, :ring]));

julia> QualityDiversity.evaluate(labelled, [1.0, 2.0]).descriptor
(2, :ring)
```
"""
struct QDProblem{F,I,S} <: AbstractQDProblem
    f::F
    init::I
    # A tuple of `(lo, hi)` pairs, or a `DescriptorCategories`.
    space::S
end

struct DescriptorCategories{T<:Tuple}
    values::T
end

# The descriptor space of `QDProblem` and `StochasticQDProblem`: a tuple of `(lo, hi)`
# pairs, or a `DescriptorCategories`.
function descriptor_space(bounds, categories)
    (bounds === nothing) == (categories === nothing) &&
        throw(ArgumentError("give exactly one of bounds and categories"))
    bounds === nothing && return DescriptorCategories(Tuple(categories))
    isempty(bounds) && throw(ArgumentError("bounds must have one (lo, hi) pair per descriptor dimension"))
    normalized = map(bounds) do b
        length(b) == 2 ||
            throw(ArgumentError("each bound must be a (lo, hi) pair, got $b"))
        (Float64(b[1]), Float64(b[2]))
    end
    return Tuple(normalized)
end

QDProblem(f; init, bounds=nothing, categories=nothing) =
    QDProblem(f, init, descriptor_space(bounds, categories))

function make_evaluation(space::Tuple, fitness, descriptor, phenotype=nothing)
    N = length(space)
    length(descriptor) == N ||
        throw(ArgumentError("descriptor has $(length(descriptor)) dimensions but the problem bounds have $N"))
    return Evaluation(fitness, SVector{N,Float64}(descriptor), phenotype)
end

function make_evaluation(space::DescriptorCategories, fitness, descriptor, phenotype=nothing)
    N = length(space.values)
    length(descriptor) == N ||
        throw(ArgumentError("descriptor has $(length(descriptor)) dimensions but the problem categories have $N"))
    return Evaluation(fitness, Tuple(descriptor), phenotype)
end

# The evaluation of what the function of a `QDProblem` or `StochasticQDProblem` returned.
evaluation_of(space, result::Tuple{Any,Any}) = make_evaluation(space, result...)
evaluation_of(space, result::Tuple{Any,Any,Any}) = make_evaluation(space, result...)

function evaluation_of(space, result)
    throw(ArgumentError("the function of the problem must return a (fitness, descriptor) or (fitness, descriptor, phenotype) tuple, got a $(typeof(result))"))
end

evaluate(prob::QDProblem, x) = evaluation_of(prob.space, prob.f(x))

# In a device kernel the descriptor stays as the function returned it; the kernel
# writes its components into the descriptor matrix of the batch.
function device_evaluate(prob::QDProblem, x)
    result = prob.f(x)
    return Evaluation(result[1], result[2])
end

random_solution(rng::AbstractRNG, prob::QDProblem) = prob.init(rng)

descriptor_bounds(prob::QDProblem{F,I,<:Tuple}) where {F,I} = prob.space

descriptor_categories(prob::QDProblem{F,I,<:DescriptorCategories}) where {F,I} = prob.space.values
