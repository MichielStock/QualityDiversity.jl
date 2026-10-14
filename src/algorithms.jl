"""
    MAPElites(prob; dims, bounds=descriptor_bounds(prob), batch_size=64,
              variation=GaussianMutation(0.1), genotype=Vector{Float64},
              fitness=Float64, outside=ThrowOutside()) -> QDAlgorithm

MAP-Elites: an archive laid out by the problem's descriptor space, and one
[`VariationEmitter`](@ref) that applies the `variation` operator to
`batch_size` children per round, with parents chosen uniformly from the
archive. `variation` can be a mutation such as [`GaussianMutation`](@ref) or a
crossover such as [`IsoLineVariation`](@ref).

For a problem with `descriptor_bounds`, the archive is a [`GridArchive`](@ref)
with `dims` cells along each descriptor dimension, laid out over `bounds`; `dims`
is required. For a problem with
[`descriptor_categories`](@ref QualityDiversity.descriptor_categories), it is a
[`CategoricalArchive`](@ref) with one cell for each combination of categories;
`dims` and `bounds` do not apply and are an error. `outside` is passed to the
archive; see [`GridArchive`](@ref) and [`CategoricalArchive`](@ref).

# Example

```julia
prob = Rastrigin(10)
result = solve(prob, MAPElites(prob; dims=(20, 20)); budget=100_000, rng=Xoshiro(1))
coverage(result.archive)
```
"""
function MAPElites(prob::AbstractQDProblem; dims::Union{Tuple,Nothing}=nothing,
                   bounds::Union{Tuple,Nothing}=nothing,
                   batch_size::Integer=64,
                   variation::AbstractVariation=GaussianMutation(0.1),
                   genotype::Type=Vector{Float64},
                   fitness::Type{<:Real}=Float64,
                   outside::OutsidePolicy=ThrowOutside())
    archive = default_archive(Val(hasmethod(descriptor_categories, Tuple{typeof(prob)})),
                              prob, dims, bounds, genotype, fitness, outside)
    return QDAlgorithm(archive, (VariationEmitter(variation; batch_size),))
end

function default_archive(::Val{false}, prob, dims, bounds, genotype, fitness, outside)
    dims === nothing &&
        throw(ArgumentError("dims is required for a problem with descriptor_bounds: pass the number of cells along each dimension, e.g. dims=(20, 20)"))
    return GridArchive(prob; dims, bounds=something(bounds, descriptor_bounds(prob)),
                       genotype, fitness, outside)
end

function default_archive(::Val{true}, prob, dims, bounds, genotype, fitness, outside)
    (dims === nothing && bounds === nothing) ||
        throw(ArgumentError("dims and bounds do not apply to a problem with descriptor_categories; its cells are the combinations of categories"))
    return CategoricalArchive(prob; genotype, fitness, outside)
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
