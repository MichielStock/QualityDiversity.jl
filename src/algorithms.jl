"""
    MAPElites(prob; dims, bounds=descriptor_bounds(prob), batch_size=64,
              variation=GaussianMutation(0.1), genotype=Vector{Float64},
              fitness=Float64, outside=ThrowOutside()) -> QDAlgorithm

MAP-Elites: a [`GridArchive`](@ref) with `dims` cells along each descriptor
dimension, laid out over `bounds`, and one
[`VariationEmitter`](@ref) that applies the `variation` operator to
`batch_size` children per round, with parents chosen uniformly from the
archive. `variation` can be a mutation such as [`GaussianMutation`](@ref) or a
crossover such as [`IsoLineVariation`](@ref).

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
                   variation::AbstractVariation=GaussianMutation(0.1),
                   genotype::Type=Vector{Float64},
                   fitness::Type{<:Real}=Float64,
                   outside::OutsidePolicy=ThrowOutside())
    archive = GridArchive(prob; dims, bounds, genotype, fitness, outside)
    return QDAlgorithm(archive, (VariationEmitter(variation; batch_size),))
end

"""
    MAPElites(archive::AbstractArchive; batch_size=64, variation=GaussianMutation(0.1))
        -> QDAlgorithm

MAP-Elites over any archive: `archive` with one [`VariationEmitter`](@ref) that
applies `variation` to `batch_size` children per round. Use it to run MAP-Elites
with an archive other than the grid, such as a [`CVTArchive`](@ref).

# Example

```julia
prob = Rastrigin(10)
archive = CVTArchive(Xoshiro(1), prob; cells=500)
result = solve(prob, MAPElites(archive); budget=100_000, rng=Xoshiro(2))
```
"""
function MAPElites(archive::AbstractArchive; batch_size::Integer=64,
                   variation::AbstractVariation=GaussianMutation(0.1))
    return QDAlgorithm(archive, (VariationEmitter(variation; batch_size),))
end
