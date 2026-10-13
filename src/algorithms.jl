"""
    MAPElites(prob; dims, bounds=descriptor_bounds(prob), batch_size=64,
              mutation=GaussianMutation(0.1), genotype=Vector{Float64},
              fitness=Float64, outside=ThrowOutside()) -> QDAlgorithm

MAP-Elites: a [`GridArchive`](@ref) with `dims` cells along each descriptor
dimension, laid out over `bounds`, and one
[`VariationEmitter`](@ref) that applies `mutation` to `batch_size` uniformly
chosen elites per round.

`bounds` and `outside` are passed to the archive; see [`GridArchive`](@ref).

# Example

```julia
prob = Rastrigin(10)
result = solve(prob, MAPElites(prob; dims=(20, 20)); budget=100_000, rng=Xoshiro(1))
coverage(result.archive)
```
"""
function MAPElites(prob::AbstractQDProblem; dims::Tuple,
                   bounds::Tuple=descriptor_bounds(prob),
                   batch_size::Integer=64,
                   mutation::AbstractVariation=GaussianMutation(0.1),
                   genotype::Type=Vector{Float64},
                   fitness::Type{<:Real}=Float64,
                   outside::OutsidePolicy=ThrowOutside())
    archive = GridArchive(prob; dims, bounds, genotype, fitness, outside)
    return QDAlgorithm(archive, (VariationEmitter(mutation; batch_size),))
end
