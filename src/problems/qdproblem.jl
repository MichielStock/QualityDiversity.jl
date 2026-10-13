"""
    QDProblem(f; init, bounds)

Wrap a plain function as a problem. `f(x)` returns a `(fitness, descriptor)`
tuple for the solution `x`. `init(rng)` returns one random solution. `bounds`
is a tuple with one `(lo, hi)` pair per descriptor dimension.

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
```
"""
struct QDProblem{F,I,B<:Tuple} <: AbstractQDProblem
    f::F
    init::I
    bounds::B
end

function QDProblem(f; init, bounds)
    isempty(bounds) && throw(ArgumentError("bounds must have one (lo, hi) pair per descriptor dimension"))
    normalized = map(bounds) do b
        length(b) == 2 ||
            throw(ArgumentError("each bound must be a (lo, hi) pair, got $b"))
        (Float64(b[1]), Float64(b[2]))
    end
    return QDProblem(f, init, Tuple(normalized))
end

function evaluate(prob::QDProblem{F,I,B}, x) where {F,I,B}
    fitness, descriptor = prob.f(x)
    N = fieldcount(B)
    length(descriptor) == N ||
        throw(ArgumentError("descriptor has $(length(descriptor)) dimensions but the problem bounds have $N"))
    return Evaluation(fitness, SVector{N,Float64}(descriptor))
end

random_solution(rng::AbstractRNG, prob::QDProblem) = prob.init(rng)

descriptor_bounds(prob::QDProblem) = prob.bounds
