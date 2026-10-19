"""
    MAPElites(prob; dims, bounds=descriptor_bounds(prob), batch_size=64,
              variation=GaussianMutation(0.1), genotype=Vector{Float64},
              fitness=Float64, outside=ThrowOutside(), phenotype=Nothing,
              device=nothing) -> QDAlgorithm

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
`dims` and `bounds` do not apply and are an error. `outside` and `phenotype` are
passed to the archive; see [`GridArchive`](@ref) and [`CategoricalArchive`](@ref).
A problem that returns a phenotype needs `phenotype` set to its type.

Everything runs on the host unless `device` names an array type, such as `CuArray`:
then the archive is a [`DeviceGridArchive`](@ref) in arrays of that type and the
emitter a [`DeviceVariationEmitter`](@ref), so the variation, the evaluation and the
insertion of a batch stay on the device. That needs a problem with `descriptor_bounds`,
real-vector genotypes, no phenotype, and a `variation` that the device emitter supports.

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
                   outside::OutsidePolicy=ThrowOutside(),
                   phenotype::Type=Nothing,
                   device::Union{Type,Nothing}=nothing)
    return map_elites(device, prob, dims, bounds, batch_size, variation, genotype, fitness, outside, phenotype)
end

function map_elites(::Nothing, prob, dims, bounds, batch_size, variation, genotype, fitness, outside, phenotype)
    archive = default_archive(Val(hasmethod(descriptor_categories, Tuple{typeof(prob)})),
                              prob, dims, bounds, genotype, fitness, outside, phenotype)
    return QDAlgorithm(archive, (VariationEmitter(variation; batch_size),))
end

function map_elites(device::Type, prob, dims, bounds, batch_size, variation, genotype, fitness, outside, phenotype)
    hasmethod(descriptor_categories, Tuple{typeof(prob)}) &&
        throw(ArgumentError("a DeviceGridArchive needs a problem with descriptor_bounds; one with descriptor_categories runs on the host"))
    dims === nothing &&
        throw(ArgumentError("dims is required for a problem with descriptor_bounds: pass the number of cells along each dimension, e.g. dims=(20, 20)"))
    phenotype === Nothing ||
        throw(ArgumentError("a DeviceGridArchive stores no phenotypes; build it without phenotype"))
    archive = DeviceGridArchive(prob; dims, device, bounds=something(bounds, descriptor_bounds(prob)),
                                genotype, fitness, outside)
    return QDAlgorithm(archive, (DeviceVariationEmitter(variation; batch_size),))
end

function default_archive(::Val{false}, prob, dims, bounds, genotype, fitness, outside, phenotype)
    dims === nothing &&
        throw(ArgumentError("dims is required for a problem with descriptor_bounds: pass the number of cells along each dimension, e.g. dims=(20, 20)"))
    return GridArchive(prob; dims, bounds=something(bounds, descriptor_bounds(prob)),
                       genotype, fitness, outside, phenotype)
end

function default_archive(::Val{true}, prob, dims, bounds, genotype, fitness, outside, phenotype)
    (dims === nothing && bounds === nothing) ||
        throw(ArgumentError("dims and bounds do not apply to a problem with descriptor_categories; its cells are the combinations of categories"))
    return CategoricalArchive(prob; genotype, fitness, outside, phenotype)
end

"""
    MAPElites(archive::AbstractArchive; batch_size=64, variation=GaussianMutation(0.1))
        -> QDAlgorithm

MAP-Elites over any archive: `archive` with one [`VariationEmitter`](@ref) that
applies `variation` to `batch_size` children per round, or a
[`DeviceVariationEmitter`](@ref) for a [`DeviceCellArchive`](@ref). Use it to run MAP-Elites
with an archive other than the grid, such as a [`CVTArchive`](@ref) or a
[`DeviceCVTArchive`](@ref).

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

function MAPElites(archive::DeviceCellArchive; batch_size::Integer=64,
                   variation::AbstractVariation=GaussianMutation(0.1))
    return QDAlgorithm(archive, (DeviceVariationEmitter(variation; batch_size),))
end

"""
    CMAME(prob; dims, sigma0, num_emitters=5, batch_size=<CMA-ES default>,
          ranker=TwoStageImprovementRanker(), selection=FilterSelection(),
          restart=NoImprovementRestart(), x0=nothing, lower=-Inf, upper=Inf,
          bounds=descriptor_bounds(prob), genotype=Vector{Float64}, fitness=Float64,
          outside=ThrowOutside(), phenotype=Nothing, device=nothing) -> QDAlgorithm

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

With `device`, an array type such as `CuArray`, the archive is a [`DeviceGridArchive`](@ref)
and the emitters are the searches of one [`DeviceCMAEmitter`](@ref): all searches ask
together, and their candidates are evaluated and added on the device as one batch, so
each search sees the archive as it was before the round. That pays off with many
emitters, hundreds rather than five.

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
               outside::OutsidePolicy=ThrowOutside(), phenotype::Type=Nothing,
               device::Union{Type,Nothing}=nothing, kwargs...)
    archive = cma_me_archive(device, prob, dims, bounds, genotype, fitness, outside, phenotype)
    return CMAME(archive; sigma0, dim=solution_dim(prob), kwargs...)
end

cma_me_archive(::Nothing, prob, dims, bounds, genotype, fitness, outside, phenotype) =
    GridArchive(prob; dims, bounds, genotype, fitness, outside, phenotype)

function cma_me_archive(device::Type, prob, dims, bounds, genotype, fitness, outside, phenotype)
    phenotype === Nothing ||
        throw(ArgumentError("a DeviceGridArchive stores no phenotypes; build it without phenotype"))
    return DeviceGridArchive(prob; dims, device, bounds, genotype, fitness, outside)
end

"""
    CMAME(archive::AbstractArchive; sigma0, dim, num_emitters=5, kwargs...) -> QDAlgorithm

CMA-ME over any archive, such as a [`CVTArchive`](@ref): `archive` with `num_emitters`
[`CMAEmitter`](@ref)s for solutions of length `dim`. The keyword arguments are those of
[`CMAEmitter`](@ref). For a [`DeviceCellArchive`](@ref) the emitters are the `num_emitters`
searches of one [`DeviceCMAEmitter`](@ref).
"""
function CMAME(archive::AbstractArchive; sigma0::Real, dim::Integer, num_emitters::Integer=5, kwargs...)
    num_emitters >= 1 || throw(ArgumentError("num_emitters must be at least 1, got $num_emitters"))
    emitters = Tuple(CMAEmitter(dim; sigma0, kwargs...) for _ in 1:num_emitters)
    return QDAlgorithm(archive, emitters)
end

function CMAME(archive::DeviceCellArchive; sigma0::Real, dim::Integer, num_emitters::Integer=5, kwargs...)
    return QDAlgorithm(archive, (DeviceCMAEmitter(dim; num_searches=num_emitters, sigma0, kwargs...),))
end

"""
    CMAMAE(prob; dims, sigma0, threshold_min, learning_rate=0.01, num_emitters=5,
           batch_size=<CMA-ES default>, ranker=ImprovementRanker(),
           selection=MuSelection(), restart=ConvergenceRestart(), x0=nothing,
           lower=-Inf, upper=Inf, bounds=descriptor_bounds(prob),
           genotype=Vector{Float64}, fitness=Float64, outside=ThrowOutside(),
           phenotype=Nothing) -> QDAlgorithm

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
                outside::OutsidePolicy=ThrowOutside(), phenotype::Type=Nothing, kwargs...)
    grid = GridArchive(prob; dims, bounds, genotype, fitness, outside, phenotype)
    archive = ThresholdArchive(grid; learning_rate, threshold_min)
    return CMAME(archive; sigma0, dim=solution_dim(prob), num_emitters, ranker, selection, restart, kwargs...)
end

"""
    DDS(rng::AbstractRNG, prob; dims, sigma0, bandwidth, buffer_size=10_000, num_emitters=15,
        batch_size=36, ranker=ImprovementRanker(), selection=MuSelection(),
        restart=ConvergenceRestart(), x0=nothing, lower=-Inf, upper=Inf,
        bounds=descriptor_bounds(prob), genotype=Vector{Float64}, fitness=Float64,
        outside=ClampOutside(), phenotype=Nothing) -> QDAlgorithm

Density Descent Search with a kernel density estimate (DDS-KDE; Lee et al., 2024): a
[`DensityArchive`](@ref) around a [`GridArchive`](@ref) with `dims` cells, and
`num_emitters` [`CMAEmitter`](@ref)s that rank their candidates by the density of the
descriptors found so far, lowest first, recombine the best half and restart on
convergence, the settings of the paper. It searches for diversity alone: the fitness only
decides which solution the grid keeps in a cell. `rng` drives the buffer of the
[`DensityArchive`](@ref), and `bandwidth` and `buffer_size` are its arguments. The grid
clamps descriptors outside `bounds` to the nearest cell by default, since the search
moves away from the descriptors it knows. The other arguments are those of
[`CMAME`](@ref).

# Example

```julia
prob = Rastrigin(10)
alg = DDS(Xoshiro(1), prob; dims=(20, 20), sigma0=0.5, bandwidth=0.5)
result = solve(prob, alg; budget=50_000, rng=Xoshiro(2))
coverage(result.archive)
```
"""
function DDS(rng::AbstractRNG, prob::AbstractQDProblem; dims::Tuple, sigma0::Real, bandwidth::Real,
             buffer_size::Integer=10_000, num_emitters::Integer=15, batch_size::Integer=36,
             ranker::AbstractRanker=ImprovementRanker(),
             selection::SelectionRule=MuSelection(),
             restart::RestartRule=ConvergenceRestart(),
             bounds::Tuple=descriptor_bounds(prob),
             genotype::Type=Vector{Float64},
             fitness::Type{<:AbstractFloat}=Float64,
             outside::OutsidePolicy=ClampOutside(), phenotype::Type=Nothing, kwargs...)
    grid = GridArchive(prob; dims, bounds, genotype, fitness, outside, phenotype)
    archive = DensityArchive(rng, grid; bandwidth, buffer_size)
    return CMAME(archive; sigma0, dim=solution_dim(prob), num_emitters, batch_size, ranker, selection,
                 restart, kwargs...)
end

"""
    CMAMEGA(prob; dims, sigma0, optimizer=AdamAscent(), num_emitters=1,
            batch_size=<CMA-ES default>, ranker=TwoStageImprovementRanker(),
            selection=FilterSelection(), restart=NoImprovementRestart(), x0=nothing,
            normalize=true, bounds=descriptor_bounds(prob), genotype=Vector{Float64},
            fitness=Float64, outside=ThrowOutside(), phenotype=Nothing) -> QDAlgorithm

CMA-MEGA (Fontaine and Nikolaidis, 2021, "Differentiable Quality Diversity"): a
[`GridArchive`](@ref) with `dims` cells along each descriptor dimension, as for
[`MAPElites`](@ref), and `num_emitters` [`GradientArborescenceEmitter`](@ref)s, which step
along the gradients of the fitness and the descriptor. The problem must provide them; see
[`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient). The defaults are those of
CMA-MEGA (Adam) in the paper, which runs one emitter.

`sigma0` is the initial step size of the distribution of the gradient coefficients and has
no default; the paper uses `10` on the sphere and Rastrigin benchmarks. The other arguments
are those of [`MAPElites`](@ref) and [`GradientArborescenceEmitter`](@ref).

# Example

```julia
prob = Sphere(100)    # gradients in closed form
result = solve(prob, CMAMEGA(prob; dims=(100, 100), sigma0=10.0); budget=20_000, rng=Xoshiro(1))
coverage(result.archive)
```
"""
function CMAMEGA(prob::AbstractQDProblem; dims::Tuple, sigma0::Real,
                 bounds::Tuple=descriptor_bounds(prob),
                 genotype::Type=Vector{Float64},
                 fitness::Type{<:Real}=Float64,
                 outside::OutsidePolicy=ThrowOutside(), phenotype::Type=Nothing, kwargs...)
    archive = GridArchive(prob; dims, bounds, genotype, fitness, outside, phenotype)
    return CMAMEGA(archive; sigma0, dim=solution_dim(prob), kwargs...)
end

"""
    CMAMEGA(archive::AbstractArchive; sigma0, dim, num_emitters=1, kwargs...) -> QDAlgorithm

CMA-MEGA over any archive with `SVector` descriptors, such as a [`CVTArchive`](@ref):
`archive` with `num_emitters` [`GradientArborescenceEmitter`](@ref)s for solutions of length
`dim`. The keyword arguments are those of [`GradientArborescenceEmitter`](@ref).
"""
function CMAMEGA(archive::AbstractArchive{G,F,SVector{N,Float64}}; sigma0::Real, dim::Integer,
                 num_emitters::Integer=1, kwargs...) where {G,F,N}
    num_emitters >= 1 || throw(ArgumentError("num_emitters must be at least 1, got $num_emitters"))
    emitters = Tuple(GradientArborescenceEmitter(dim; num_descriptors=N, sigma0, kwargs...) for _ in 1:num_emitters)
    return QDAlgorithm(archive, emitters)
end

function CMAMEGA(archive::AbstractArchive; kwargs...)
    throw(ArgumentError("differentiable quality diversity needs an archive with SVector descriptors, not a $(nameof(typeof(archive)))"))
end

"""
    CMAMAEGA(prob; dims, sigma0, threshold_min, learning_rate=0.01, optimizer=AdamAscent(),
             num_emitters=1, batch_size=<CMA-ES default>, ranker=ImprovementRanker(),
             selection=MuSelection(), restart=ConvergenceRestart(), x0=nothing,
             normalize=true, bounds=descriptor_bounds(prob), genotype=Vector{Float64},
             fitness=Float64, outside=ThrowOutside(), phenotype=Nothing) -> QDAlgorithm

CMA-MAEGA (Fontaine and Nikolaidis, 2023): [`CMAMEGA`](@ref) with the rules of
[`CMAMAE`](@ref): candidates ranked by their improvement over the threshold of their cell,
the best half recombined, restarts on convergence only, over a [`ThresholdArchive`](@ref)
around a [`GridArchive`](@ref). `threshold_min` and `learning_rate` are those of the
[`ThresholdArchive`](@ref); `optimizer`, the rule that moves the solution point, is that of
the [`GradientArborescenceEmitter`](@ref)s. `solve` returns the result archive, which holds
the best solution of every cell.
"""
function CMAMAEGA(prob::AbstractQDProblem; dims::Tuple, sigma0::Real, threshold_min::Real,
                  learning_rate::Real=0.01,
                  ranker::AbstractRanker=ImprovementRanker(),
                  selection::SelectionRule=MuSelection(),
                  restart::RestartRule=ConvergenceRestart(),
                  bounds::Tuple=descriptor_bounds(prob),
                  genotype::Type=Vector{Float64},
                  fitness::Type{<:AbstractFloat}=Float64,
                  outside::OutsidePolicy=ThrowOutside(), phenotype::Type=Nothing, kwargs...)
    grid = GridArchive(prob; dims, bounds, genotype, fitness, outside, phenotype)
    archive = ThresholdArchive(grid; learning_rate, threshold_min)
    return CMAMEGA(archive; sigma0, dim=solution_dim(prob), ranker, selection, restart, kwargs...)
end

"""
    OMGMEGA(prob; dims, sigma, batch_size=64, normalize=true, bounds=descriptor_bounds(prob),
            genotype=Vector{Float64}, fitness=Float64, outside=ThrowOutside(),
            phenotype=Nothing) -> QDAlgorithm

OMG-MEGA (Fontaine and Nikolaidis, 2021, "Differentiable Quality Diversity"): MAP-Elites
whose children are gradient steps from their parents, a [`GradientArchive`](@ref) around a
[`GridArchive`](@ref) with `dims` cells along each descriptor dimension and a
[`GradientOperatorEmitter`](@ref). Every solution of the run is evaluated with its
gradients; see [`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient). `sigma`, the
standard deviation of the gradient coefficients, has no default; the paper uses `10` on the
sphere and Rastrigin benchmarks. The other arguments are those of [`MAPElites`](@ref) and
[`GradientOperatorEmitter`](@ref).
"""
function OMGMEGA(prob::AbstractQDProblem; dims::Tuple, sigma::Real,
                 bounds::Tuple=descriptor_bounds(prob),
                 genotype::Type=Vector{Float64},
                 fitness::Type{<:Real}=Float64,
                 outside::OutsidePolicy=ThrowOutside(), phenotype::Type=Nothing, kwargs...)
    archive = GradientArchive(GridArchive(prob; dims, bounds, genotype, fitness, outside, phenotype))
    return OMGMEGA(archive; sigma, kwargs...)
end

"""
    OMGMEGA(archive::GradientArchive; sigma, batch_size=64, normalize=true) -> QDAlgorithm

OMG-MEGA over any [`GradientArchive`](@ref), such as one around a [`CVTArchive`](@ref).
"""
OMGMEGA(archive::GradientArchive; sigma::Real, batch_size::Integer=64, normalize::Bool=true) =
    QDAlgorithm(archive, (GradientOperatorEmitter(; sigma, batch_size, normalize),))

# Length of the solutions of `prob`, from one random solution.
solution_dim(prob::AbstractQDProblem) = length(random_solution(Random.Xoshiro(0), prob))

"""
    MOME(prob; dims, reference_point=reference_point(prob), max_front=50, batch_size=64,
         variation=GaussianMutation(0.1), bounds=descriptor_bounds(prob),
         genotype=Vector{Float64}, outside=ThrowOutside(), duplicates=RejectDuplicates(),
         full_front=RejectNewcomer(), phenotype=Nothing) -> QDAlgorithm

Multi-objective MAP-Elites (Pierrot et al., 2022): a [`ParetoGridArchive`](@ref), which keeps a
Pareto front of at most `max_front` solutions in each cell of a grid with `dims` cells along
each descriptor dimension, and one [`VariationEmitter`](@ref) that applies `variation` to
`batch_size` children per round. A parent is a solution drawn uniformly from the front of a
cell drawn uniformly among the occupied ones. The problem returns a vector of objectives as
its fitness. `reference_point` is the corner from which hypervolume is measured; a problem
can supply it with [`reference_point`](@ref QualityDiversity.reference_point). `duplicates` and
`full_front` decide about a solution whose objectives equal those of a stored one and about
one for a full front; see [`ParetoGridArchive`](@ref). With `full_front=ReplaceCrowded()` the
archive keeps fronts as MOME-PGX does (Janmohamed et al., 2023), whose crowding-based
selection of parents and policy-gradient variation are not part of this constructor.

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
              outside::OutsidePolicy=ThrowOutside(), duplicates::DuplicatePolicy=RejectDuplicates(),
              full_front::FullFrontPolicy=RejectNewcomer(), phenotype::Type=Nothing)
    archive = ParetoGridArchive(prob; dims, reference_point, max_front, bounds, genotype, outside,
                                duplicates, full_front, phenotype)
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

"""
    MELS(prob::SampledSpread; dims, batch_size=64, variation=GaussianMutation(0.1),
         bounds=descriptor_bounds(prob), genotype=Vector{Float64}, fitness=Float64,
         outside=ThrowOutside(), phenotype=Nothing) -> QDAlgorithm

MAP-Elites with low spread (Mace et al., 2023): MAP-Elites for a noisy problem in which an
elite is replaced only by a solution with higher fitness and no larger spread, the mean
distance between the descriptors of its evaluations. The archive is a
[`SpreadArchive`](@ref) around a [`GridArchive`](@ref) with `dims` cells along each
descriptor dimension, and one [`VariationEmitter`](@ref) applies `variation` to
`batch_size` children per round.

`prob` is a [`SampledSpread`](@ref) problem, which evaluates every solution `num_samples`
times; pass the same problem to `solve`. The other arguments are those of
[`MAPElites`](@ref).

# Example

```julia
noisy = StochasticQDProblem((rng, x) -> (-sum(abs2, x), (x[1], x[2]) .+ 0.05 .* randn(rng, 2));
                            init=rng -> randn(rng, 5), bounds=((-3, 3), (-3, 3)))
prob = SampledSpread(noisy; num_samples=10)
result = solve(prob, MELS(prob; dims=(20, 20)); budget=200_000, rng=Xoshiro(1))
```
"""
function MELS(prob::SampledSpread; dims::Tuple, batch_size::Integer=64,
              variation::AbstractVariation=GaussianMutation(0.1),
              bounds::Tuple=descriptor_bounds(prob), genotype::Type=Vector{Float64},
              fitness::Type{<:Real}=Float64, outside::OutsidePolicy=ThrowOutside(),
              phenotype::Type=Nothing)
    archive = SpreadArchive(GridArchive(prob; dims, bounds, genotype, fitness, outside, phenotype))
    return QDAlgorithm(archive, (VariationEmitter(variation; batch_size),))
end

function MELS(prob::AbstractQDProblem; kwargs...)
    throw(ArgumentError("MELS needs a SampledSpread problem, which evaluates every solution several times: wrap the stochastic problem as SampledSpread(prob; num_samples)"))
end

"""
    MELS(archive::SpreadArchive; batch_size=64, variation=GaussianMutation(0.1)) -> QDAlgorithm

MELS over a given archive, such as a [`SpreadArchive`](@ref) around a [`CVTArchive`](@ref):
`archive` with one [`VariationEmitter`](@ref).
"""
function MELS(archive::SpreadArchive; batch_size::Integer=64,
              variation::AbstractVariation=GaussianMutation(0.1))
    return QDAlgorithm(archive, (VariationEmitter(variation; batch_size),))
end
