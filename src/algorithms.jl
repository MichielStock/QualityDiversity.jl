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

"""
    CMAME(prob; dims, sigma0, num_emitters=5, batch_size=<CMA-ES default>,
          ranker=TwoStageImprovementRanker(), selection=FilterSelection(),
          restart=NoImprovementRestart(), x0=nothing, lower=-Inf, upper=Inf,
          bounds=descriptor_bounds(prob), genotype=Vector{Float64}, fitness=Float64,
          outside=ThrowOutside()) -> QDAlgorithm

CMA-ME (Fontaine et al., 2020): a [`GridArchive`](@ref) with `dims` cells along each
descriptor dimension, as for [`MAPElites`](@ref), and `num_emitters` independent
[`CMAEmitter`](@ref)s, each with its own search distribution. The defaults are the
improvement emitter of the paper: candidates ranked by what the archive did with them,
the ones the archive accepted recombined, and a restart from a random elite when a
batch is rejected entirely.

`sigma0` is the initial step size of the search distributions and has no default,
because it depends on the scale of the solutions. The solutions must be real vectors;
their length is that of `random_solution(rng, prob)`. `lower` and `upper` bound the
solutions, and `x0` gives the mean of the first distribution of every emitter, by
default a random elite. The other arguments are those of [`MAPElites`](@ref) and
[`CMAEmitter`](@ref).

# Example

```julia
prob = Rastrigin(10)
result = solve(prob, CMAME(prob; dims=(20, 20), sigma0=1.0); budget=50_000, rng=Xoshiro(1))
qd_score(result.archive; offset=fitness_offset(prob))
```
"""
function CMAME(prob::AbstractQDProblem; dims::Tuple, sigma0::Real,
               bounds::Tuple=descriptor_bounds(prob),
               genotype::Type=Vector{Float64},
               fitness::Type{<:Real}=Float64,
               outside::OutsidePolicy=ThrowOutside(), kwargs...)
    archive = GridArchive(prob; dims, bounds, genotype, fitness, outside)
    return CMAME(archive; sigma0, dim=solution_dim(prob), kwargs...)
end

"""
    CMAME(archive::AbstractArchive; sigma0, dim, num_emitters=5, kwargs...) -> QDAlgorithm

CMA-ME over any archive, such as a [`CVTArchive`](@ref): `archive` with `num_emitters`
[`CMAEmitter`](@ref)s for solutions of length `dim`. The keyword arguments are those of
[`CMAEmitter`](@ref).
"""
function CMAME(archive::AbstractArchive; sigma0::Real, dim::Integer, num_emitters::Integer=5, kwargs...)
    num_emitters >= 1 || throw(ArgumentError("num_emitters must be at least 1, got $num_emitters"))
    emitters = Tuple(CMAEmitter(dim; sigma0, kwargs...) for _ in 1:num_emitters)
    return QDAlgorithm(archive, emitters)
end

"""
    CMAMAE(prob; dims, sigma0, threshold_min, learning_rate=0.01, num_emitters=5,
           batch_size=<CMA-ES default>, ranker=ImprovementRanker(),
           selection=MuSelection(), restart=ConvergenceRestart(), x0=nothing,
           lower=-Inf, upper=Inf, bounds=descriptor_bounds(prob),
           genotype=Vector{Float64}, fitness=Float64, outside=ThrowOutside())
        -> QDAlgorithm

CMA-MAE (Fontaine and Nikolaidis, 2023): [`CMAEmitter`](@ref)s as for [`CMAME`](@ref)
but ranking candidates by their improvement over the threshold of their cell,
recombining the best half, and restarting only on convergence, over a
[`ThresholdArchive`](@ref) around a [`GridArchive`](@ref). `threshold_min`, the
lowest fitness the problem can produce, and `learning_rate` are those of the
[`ThresholdArchive`](@ref). Because the elites of that archive can get worse, `solve`
returns its result archive, which holds the best solution of every cell.
"""
function CMAMAE(prob::AbstractQDProblem; dims::Tuple, sigma0::Real, threshold_min::Real,
                learning_rate::Real=0.01, num_emitters::Integer=5,
                ranker::AbstractRanker=ImprovementRanker(),
                selection::SelectionRule=MuSelection(),
                restart::RestartRule=ConvergenceRestart(),
                bounds::Tuple=descriptor_bounds(prob),
                genotype::Type=Vector{Float64},
                fitness::Type{<:AbstractFloat}=Float64,
                outside::OutsidePolicy=ThrowOutside(), kwargs...)
    grid = GridArchive(prob; dims, bounds, genotype, fitness, outside)
    archive = ThresholdArchive(grid; learning_rate, threshold_min)
    return CMAME(archive; sigma0, dim=solution_dim(prob), num_emitters, ranker, selection, restart, kwargs...)
end

# Length of the solutions of `prob`, from one random solution.
solution_dim(prob::AbstractQDProblem) = length(random_solution(Random.Xoshiro(0), prob))

"""
    MOME(prob; dims, reference_point=reference_point(prob), max_front=50, batch_size=64,
         variation=GaussianMutation(0.1), bounds=descriptor_bounds(prob),
         genotype=Vector{Float64}, outside=ThrowOutside()) -> QDAlgorithm

Multi-objective MAP-Elites (Pierrot et al., 2022): a [`ParetoGridArchive`](@ref), which keeps a
Pareto front of at most `max_front` solutions in each cell of a grid with `dims` cells along
each descriptor dimension, and one [`VariationEmitter`](@ref) that applies `variation` to
`batch_size` children per round. A parent is a solution drawn uniformly from the front of a
cell drawn uniformly among the occupied ones. The problem returns a vector of objectives as
its fitness. `reference_point` is the corner from which hypervolume is measured; a problem
can supply it with [`reference_point`](@ref QualityDiversity.reference_point).

`solve` returns the archive, and a run records its MOQD score, coverage, the largest
hypervolume of a cell and the number of solutions.

# Example

```julia
prob = BiSphere(10)
result = solve(prob, MOME(prob; dims=(20, 20)); budget=50_000, rng=Xoshiro(1))
qd_score(result.archive), global_hypervolume(result.archive)
```
"""
# The reference point of the problem, as a default for the keyword of the same name.
problem_reference_point(prob) = reference_point(prob)

function MOME(prob::AbstractQDProblem; dims::Tuple,
              reference_point::AbstractVector{<:Real}=problem_reference_point(prob),
              max_front::Integer=50, batch_size::Integer=64,
              variation::AbstractVariation=GaussianMutation(0.1),
              bounds::Tuple=descriptor_bounds(prob), genotype::Type=Vector{Float64},
              outside::OutsidePolicy=ThrowOutside())
    archive = ParetoGridArchive(prob; dims, reference_point, max_front, bounds, genotype, outside)
    return QDAlgorithm(archive, (VariationEmitter(variation; batch_size),))
end

"""
    MOME(archive::ParetoGridArchive; batch_size=64, variation=GaussianMutation(0.1)) -> QDAlgorithm

MOME over a given archive: `archive` with one [`VariationEmitter`](@ref).
"""
function MOME(archive::ParetoGridArchive; batch_size::Integer=64,
              variation::AbstractVariation=GaussianMutation(0.1))
    return QDAlgorithm(archive, (VariationEmitter(variation; batch_size),))
end
