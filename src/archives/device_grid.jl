"""
    DeviceBatch(X::AbstractMatrix)

A batch of real-vector solutions held as the columns of the matrix `X`, which may be
a device array. It is an `AbstractVector` whose `j`th element is `view(X, :, j)`;
`parent(batch)` returns `X`. A [`DeviceVariationEmitter`](@ref) returns its candidates
as a `DeviceBatch`, and [`evaluate_batch`](@ref QualityDiversity.evaluate_batch) and
[`add_batch!`](@ref QualityDiversity.add_batch!) of a [`DeviceGridArchive`](@ref) take
it without copying it to the host. A problem that scores a batch with array
operations defines `evaluate_batch(prob, xs::DeviceBatch)` and works on `parent(xs)`.
"""
struct DeviceBatch{T,M<:AbstractMatrix{T},E} <: AbstractVector{E}
    X::M
    function DeviceBatch(X::M) where {T,M<:AbstractMatrix{T}}
        Base.require_one_based_indexing(X)
        # A GPU package may return a contiguous view as its own array type.
        return new{T,M,Base.promote_op(view, M, Colon, Int)}(X)
    end
end

Base.parent(batch::DeviceBatch) = batch.X
Base.size(batch::DeviceBatch) = (size(batch.X, 2),)
Base.getindex(batch::DeviceBatch, j::Int) = view(batch.X, :, j)

"""
    EvaluationBatch{N}(fitness::AbstractVector, descriptors::AbstractMatrix)

The evaluations of a batch of solutions as two arrays, which may be device arrays:
the fitness of solution `j` is `fitness[j]` and its descriptor the column
`descriptors[:, j]`, with `N` rows. It is an `AbstractVector` of [`Evaluation`](@ref)s
with `SVector{N,Float64}` descriptors and no phenotype; reading one element copies it
to the host. The default [`evaluate_batch`](@ref QualityDiversity.evaluate_batch) of a
[`DeviceBatch`](@ref) returns one, and [`add_batch!`](@ref QualityDiversity.add_batch!)
of a [`DeviceGridArchive`](@ref) reads its arrays where they are.
"""
struct EvaluationBatch{N,F,V<:AbstractVector{F},M<:AbstractMatrix} <: AbstractVector{Evaluation{F,SVector{N,Float64},Nothing}}
    fitness::V
    descriptors::M
    function EvaluationBatch{N}(fitness::V, descriptors::M) where {N,F,V<:AbstractVector{F},M<:AbstractMatrix}
        Base.require_one_based_indexing(fitness, descriptors)
        size(descriptors) == (N, length(fitness)) ||
            throw(DimensionMismatch("descriptors must be $N × $(length(fitness)), one column per solution, got $(join(size(descriptors), " × "))"))
        return new{N,F,V,M}(fitness, descriptors)
    end
end

Base.size(batch::EvaluationBatch) = size(batch.fitness)

function Base.getindex(batch::EvaluationBatch{N}, j::Int) where {N}
    fitness = only(Array(view(batch.fitness, j:j)))
    descriptor = SVector{N,Float64}(Array(view(batch.descriptors, :, j)))
    return Evaluation(fitness, descriptor)
end

"""
    AddResultBatch{F}

The results of [`add_batch!`](@ref QualityDiversity.add_batch!) on a
[`DeviceGridArchive`](@ref) for a [`DeviceBatch`](@ref): an `AbstractVector` of
[`AddResult`](@ref)s whose fields are kept in two arrays on the device, `status` (the
`AddStatus` as a `UInt8`) and `improvement`. Reading one element copies it to the host;
`collect` copies all of them.
"""
struct AddResultBatch{F,S<:AbstractVector{UInt8},V<:AbstractVector{F}} <: AbstractVector{AddResult{F}}
    status::S
    improvement::V
end

Base.size(batch::AddResultBatch) = size(batch.status)

function Base.getindex(batch::AddResultBatch, j::Int)
    return AddResult(AddStatus(only(Array(view(batch.status, j:j)))),
                     only(Array(view(batch.improvement, j:j))))
end

function Base.collect(batch::AddResultBatch{F}) where {F}
    return AddResult{F}[AddResult(AddStatus(s), i)
                        for (s, i) in zip(Array(batch.status), Array(batch.improvement))]
end

"""
    DeviceGridArchive{T,F}(device, dims, bounds, dim; outside=ThrowOutside())
    DeviceGridArchive(prob::AbstractQDProblem; dims, device, bounds=descriptor_bounds(prob),
                      genotype=Vector{Float64}, fitness=Float64, outside=ThrowOutside())

A grid archive held in the arrays of a device, for MAP-Elites on a GPU. It has the cells
and the rules of a [`GridArchive`](@ref) and stores solutions that are real vectors of
length `dim` with entries of a floating-point type `T`, and fitness values of type `F`. `device` is the
array type to store them in: `CuArray` (CUDA.jl), `ROCArray` (AMDGPU.jl), `MtlArray`
(Metal.jl), `oneArray` (oneAPI.jl), or `Array`, which runs the same kernels on the CPU.
The second form takes `T` from `genotype`, `F` from `fitness` and `dim` from
[`random_solution(rng, prob)`](@ref QualityDiversity.random_solution).

The kernels that insert solutions and that vary them in a
[`DeviceVariationEmitter`](@ref) are written with KernelAbstractions.jl, which every GPU
package above loads; with `device=Array` load it yourself (`using KernelAbstractions`).
[`MAPElites`](@ref) with `device` builds this archive and that emitter.

Storage is one array per field, on the device: the genotypes as a `dim × cells` matrix,
the descriptors as an `N × cells` matrix of `T`, the fitness values, which cells are
occupied, and the occupied cells in increasing order, from which parents are drawn
uniformly. The archive stores no phenotypes.

[`add_batch!`](@ref QualityDiversity.add_batch!) follows the rule of `GridArchive`: every
result is computed against the archive as it was before the batch, and of the solutions
that enter a cell the best is kept, the first in the batch on a tie. A fitness value that
is `NaN` throws an `ArgumentError`, as does a descriptor outside the bounds with
`ThrowOutside()` or a `NaN` component with `ClampOutside()`, before the archive changes.

The other functions of the archive contract work on the host: `add!` adds one solution
through `add_batch!`, and `elites` and `sample_elites` copy the archive to the host
first. `GridArchive(archive)` returns a host copy, for example to plot it. The metrics are
reductions on the device.

Consumer GPUs compute `Float64` many times more slowly than `Float32`, and some devices
have no `Float64` at all: use `genotype=Vector{Float32}` and `fitness=Float32` there.

# Example

```julia
using CUDA
prob = Rastrigin(20)
alg = MAPElites(prob; dims=(100, 100), batch_size=10_000, device=CuArray,
                genotype=Vector{Float32}, fitness=Float32, variation=GaussianMutation(0.5f0))
result = solve(prob, alg; budget=1_000_000, rng=Xoshiro(1))
coverage(result.archive), GridArchive(result.archive)
```
"""
mutable struct DeviceGridArchive{T<:AbstractFloat,F<:Real,N,O<:OutsidePolicy,M<:AbstractMatrix{T},V<:AbstractVector{F},
                                 B<:AbstractVector{Bool},I<:AbstractVector{Int32},K<:AbstractVector{UInt64}} <:
               AbstractArchive{Vector{T},F,SVector{N,Float64},Nothing}
    const dims::NTuple{N,Int}
    const lower::SVector{N,Float64}
    const upper::SVector{N,Float64}
    const outside::O
    const occupied::B
    const fitness::V
    const descriptors::M
    const genotypes::M
    # The occupied cells in increasing order; the first `count` entries are valid.
    const occupied_idx::I
    count::Int
    # Scratch space of `add_batch!`, one entry per cell, restored after every batch:
    # the largest fitness key that enters the cell (zero for none) and the first
    # position in the batch that has it (`typemax(Int32)` for none).
    const best_key::K
    const winner::I

    # The first solution of a batch with a descriptor the archive refuses, the first with a
    # NaN fitness (`typemax(Int32)` for none), and whether the batch fills a new cell
    # (zero for no); restored after every batch.
    const flags::I

    function DeviceGridArchive{T,F,N,O,M,V,B,I,K}(dims, lower, upper, outside, occupied, fitness, descriptors,
                                                  genotypes, occupied_idx, count, best_key,
                                                  winner, flags) where {T,F,N,O,M,V,B,I,K}
        return new{T,F,N,O,M,V,B,I,K}(dims, lower, upper, outside, occupied, fitness, descriptors, genotypes,
                                      occupied_idx, count, best_key, winner, flags)
    end
end

function DeviceGridArchive{T,F}(device::Type, dims::Tuple, bounds::Tuple, dim::Integer;
                                outside::OutsidePolicy=ThrowOutside()) where {T<:AbstractFloat,F<:Real}
    N = length(dims)
    N >= 1 || throw(ArgumentError("dims must have at least one entry"))
    length(bounds) == N ||
        throw(ArgumentError("dims has $N entries but bounds has $(length(bounds))"))
    dim >= 1 || throw(ArgumentError("solutions must have at least one entry, got dim = $dim"))
    lower = SVector{N,Float64}(map(b -> Float64(b[1]), bounds))
    upper = SVector{N,Float64}(map(b -> Float64(b[2]), bounds))
    for i in 1:N
        dims[i] >= 1 ||
            throw(ArgumentError("dims[$i] is $(dims[i]) but every dimension needs at least one cell"))
        (isfinite(lower[i]) && isfinite(upper[i]) && lower[i] < upper[i]) ||
            throw(ArgumentError("bounds[$i] is ($(lower[i]), $(upper[i])) but lo must be finite and less than hi"))
    end
    ncells = prod(dims)
    ncells < typemax(Int32) ||
        throw(ArgumentError("a DeviceGridArchive holds fewer than $(typemax(Int32)) cells, got $ncells"))
    genotypes = fill!(device{T}(undef, Int(dim), ncells), zero(T))
    descriptors = fill!(similar(genotypes, T, N, ncells), zero(T))
    fitness = fill!(similar(genotypes, F, ncells), zero(F))
    occupied = fill!(similar(genotypes, Bool, ncells), false)
    occupied_idx = fill!(similar(genotypes, Int32, ncells), Int32(0))
    best_key = fill!(similar(genotypes, UInt64, ncells), zero(UInt64))
    winner = fill!(similar(genotypes, Int32, ncells), typemax(Int32))
    flags = copyto!(similar(genotypes, Int32, 3), Int32[typemax(Int32), typemax(Int32), 0])
    return DeviceGridArchive{T,F,N,typeof(outside),typeof(genotypes),typeof(fitness),typeof(occupied),
                             typeof(occupied_idx),typeof(best_key)}(
        map(Int, dims), lower, upper, outside, occupied, fitness, descriptors, genotypes,
        occupied_idx, 0, best_key, winner, flags)
end

function DeviceGridArchive(prob::AbstractQDProblem; dims::Tuple, device::Type,
                           bounds::Tuple=descriptor_bounds(prob),
                           genotype::Type=Vector{Float64}, fitness::Type{<:Real}=Float64,
                           outside::OutsidePolicy=ThrowOutside())
    T = device_eltype(genotype)
    return DeviceGridArchive{T,fitness}(device, dims, bounds, solution_dim(prob); outside)
end

device_eltype(::Type{<:AbstractVector{T}}) where {T<:AbstractFloat} = T

function device_eltype(G::Type)
    throw(ArgumentError("a DeviceGridArchive stores vectors of floating-point numbers, so genotype must be a type such as Vector{Float32}, got $G"))
end

Base.length(archive::DeviceGridArchive) = archive.count

capacity(archive::DeviceGridArchive) = length(archive.occupied)

descriptor_bounds(archive::DeviceGridArchive) = Tuple(zip(archive.lower, archive.upper))

"The length of the solutions that `archive` stores."
solution_length(archive::DeviceGridArchive) = size(archive.genotypes, 1)

function cell_index(archive::DeviceGridArchive, d::AbstractVector{<:Real})
    return grid_cell(archive.dims, archive.lower, archive.upper, archive.outside, d)
end

function Base.empty!(archive::DeviceGridArchive)
    fill!(archive.occupied, false)
    archive.count = 0
    return archive
end

# The valid entries of `occupied_idx` on the host.
host_occupied(archive::DeviceGridArchive) = Int.(Array(view(archive.occupied_idx, 1:archive.count)))

function elites(archive::DeviceGridArchive{T,F,N}) where {T,F,N}
    cells = host_occupied(archive)
    genotypes = Array(archive.genotypes)
    fitness = Array(archive.fitness)
    descriptors = Array(archive.descriptors)
    return ((genotypes[:, i], Evaluation(fitness[i], SVector{N,Float64}(view(descriptors, :, i))))
            for i in cells)
end

"""
    sample_elites(rng, archive::DeviceGridArchive, n) -> Vector{Vector{T}}

`n` genotypes drawn with replacement, uniformly over the occupied cells, as host
vectors. A [`DeviceVariationEmitter`](@ref) draws its parents on the device instead.
"""
function sample_elites(rng::AbstractRNG, archive::DeviceGridArchive{T}, n::Integer) where {T}
    archive.count == 0 && throw(ArgumentError("cannot sample elites from an empty archive"))
    cells = host_occupied(archive)
    genotypes = Array(archive.genotypes)
    parents = Vector{Vector{T}}(undef, n)
    for k in 1:n
        parents[k] = genotypes[:, rand(rng, cells)]
    end
    return parents
end

function add!(archive::DeviceGridArchive, x, ev::Evaluation{<:Real})
    return only(add_batch!(archive, [x], [ev]))
end

"""
    add_batch!(archive::DeviceGridArchive, xs, evs)

Insert a batch with the rule of [`GridArchive`](@ref): every result is computed
against the archive as it was before the batch, and of the solutions that enter a cell
the best is kept (the first on a tie). With a [`DeviceBatch`](@ref) the solutions stay
on the device and the results are an [`AddResultBatch`](@ref) there; `evs` may be an
[`EvaluationBatch`](@ref) or a vector of [`Evaluation`](@ref)s, which is copied to the
device. Any other vector of solutions is copied to the device, and the results come back
as a `Vector{AddResult{F}}`.
"""
function add_batch!(archive::DeviceGridArchive, xs::DeviceBatch, evs::EvaluationBatch)
    return device_add_batch!(archive, parent(xs), evs.fitness, evs.descriptors)
end

function add_batch!(archive::DeviceGridArchive{T,F,N}, xs::DeviceBatch, evs::AbstractVector) where {T,F,N}
    length(xs) == length(evs) ||
        throw(ArgumentError("got $(length(xs)) solutions but $(length(evs)) evaluations"))
    return add_batch!(archive, xs, upload_evaluations(archive, evs))
end

function add_batch!(archive::DeviceGridArchive{T,F,N}, xs::AbstractVector, evs::AbstractVector) where {T,F,N}
    length(xs) == length(evs) ||
        throw(ArgumentError("got $(length(xs)) solutions but $(length(evs)) evaluations"))
    return collect(add_batch!(archive, initial_batch(archive, xs), upload_evaluations(archive, evs)))
end

function upload_evaluations(archive::DeviceGridArchive{T,F,N}, evs::EvaluationBatch) where {T,F,N}
    return evs
end

function upload_evaluations(archive::DeviceGridArchive{T,F,N}, evs::AbstractVector) where {T,F,N}
    fitness = Vector{F}(undef, length(evs))
    descriptors = Matrix{T}(undef, N, length(evs))
    for (j, ev) in zip(eachindex(fitness), evs)
        stored_phenotype(Nothing, ev.phenotype)    # a DeviceGridArchive stores no phenotypes
        length(ev.descriptor) == N ||
            throw(ArgumentError("descriptor has $(length(ev.descriptor)) dimensions but the archive has $N"))
        fitness[j] = ev.fitness
        descriptors[:, j] = ev.descriptor
    end
    return EvaluationBatch{N}(copyto!(similar(archive.fitness, F, length(fitness)), fitness),
                              copyto!(similar(archive.genotypes, T, size(descriptors)), descriptors))
end

# The default `evaluate_batch` of a device batch: `device_evaluate` in a kernel, one
# solution per thread.
function default_evaluate_batch(prob::AbstractQDProblem, xs::DeviceBatch{T}) where {T}
    X = parent(xs)
    N = length(descriptor_bounds(prob))
    # Inferred for a host vector, which takes the element-wise code that a kernel runs;
    # a device array type on the host may take a generic path that does not infer.
    E = Base.promote_op(device_evaluate, typeof(prob), Vector{T})
    F = device_fitness_type(E, prob)
    check_device_descriptor(fieldtype(E, :descriptor), N, prob)
    fitness = similar(X, F, length(xs))
    descriptors = similar(X, T, N, length(xs))
    device_evaluate!(fitness, descriptors, prob, X)
    return EvaluationBatch{N}(fitness, descriptors)
end

device_fitness_type(::Type{<:Evaluation{F}}, prob) where {F<:Real} = F

function device_fitness_type(E::Type, prob)
    throw(ArgumentError("the evaluation of $(nameof(typeof(prob))) on a device batch has type $E, not an Evaluation with a real fitness; define QualityDiversity.evaluate_batch(prob::$(nameof(typeof(prob))), xs::DeviceBatch)"))
end

# A kernel writes the descriptor component by component, so its length must be known
# from its type.
function check_device_descriptor(D::Type, N, prob)
    M = D <: Tuple && isconcretetype(D) ? fieldcount(D) :
        D <: SVector && isconcretetype(D) ? length(D) : nothing
    M === nothing &&
        throw(ArgumentError("the descriptor of $(nameof(typeof(prob))) on a device batch has type $D; it must be a tuple or an SVector"))
    M == N ||
        throw(ArgumentError("the descriptor of $(nameof(typeof(prob))) has $M dimensions but its bounds have $N"))
    return nothing
end

# The initial population is copied to the device once and evaluated there.
function initial_batch(archive::DeviceGridArchive{T}, xs::AbstractVector) where {T}
    X = Matrix{T}(undef, solution_length(archive), length(xs))
    for (j, x) in zip(axes(X, 2), xs)
        length(x) == size(X, 1) ||
            throw(ArgumentError("solution has $(length(x)) entries but the archive stores solutions of length $(size(X, 1))"))
        X[:, j] = x
    end
    return DeviceBatch(copyto!(similar(archive.genotypes, T, size(X)), X))
end

# The device kernels live in the KernelAbstractions extension; these methods are
# reached only when it is not loaded.
function device_add_batch!(archive, X, fitness, descriptors)
    throw(ArgumentError(DEVICE_KERNELS_MESSAGE))
end

device_evaluate!(fitness, descriptors, prob, X) = throw(ArgumentError(DEVICE_KERNELS_MESSAGE))

const DEVICE_KERNELS_MESSAGE = "the kernels of DeviceGridArchive need KernelAbstractions.jl: load it, or a GPU package that loads it (CUDA.jl, AMDGPU.jl, Metal.jl, oneAPI.jl)"

# Rebuilds the list of occupied cells after a batch that filled new cells. The list is
# in increasing cell order, so the parents that the emitter draws depend only on the
# archive and the seed.
function refresh_occupied!(archive::DeviceGridArchive)
    cells = findall(Array(archive.occupied))
    archive.count = length(cells)
    copyto!(archive.occupied_idx, 1, Int32.(cells), 1, length(cells))
    return nothing
end

# The sum of the fitness values of the occupied cells, accumulated in Float64, and their
# maximum, in one reduction on the device.
function fitness_summary(archive::DeviceGridArchive{T,F}) where {T,F}
    return mapreduce((o, f) -> o ? (Float64(f), f) : (0.0, typemin(F)),
                     (a, b) -> (a[1] + b[1], max(a[2], b[2])),
                     archive.occupied, archive.fitness; init=(0.0, typemin(F)))
end

qd_score(archive::DeviceGridArchive; offset=0) = fitness_summary(archive)[1] + length(archive) * offset

function max_fitness(archive::DeviceGridArchive)
    isempty_archive(archive, "max_fitness")
    return fitness_summary(archive)[2]
end

function mean_fitness(archive::DeviceGridArchive)
    isempty_archive(archive, "mean_fitness")
    return fitness_summary(archive)[1] / length(archive)
end

function metrics_record(archive::DeviceGridArchive, n_evals, offset)
    n = length(archive)
    total, maximum = fitness_summary(archive)
    return (n_evals=n_evals,
            qd_score=total + n * Float64(offset),
            coverage=n / capacity(archive),
            max_fitness=n == 0 ? NaN : Float64(maximum),
            mean_fitness=n == 0 ? NaN : total / n)
end

"""
    GridArchive(archive::DeviceGridArchive)

A [`GridArchive`](@ref) on the host with the cells, bounds, `outside` policy and elites
of `archive`, for the functions that need a host archive, such as plotting. Its
genotypes are `Vector{T}`.
"""
function GridArchive(archive::DeviceGridArchive{T,F,N}) where {T,F,N}
    host = GridArchive{Vector{T},F}(archive.dims, descriptor_bounds(archive); outside=archive.outside)
    genotypes = Array(archive.genotypes)
    fitness = Array(archive.fitness)
    descriptors = Array(archive.descriptors)
    for i in host_occupied(archive)
        place!(host, i, genotypes[:, i], SVector{N,Float64}(view(descriptors, :, i)), fitness[i], nothing)
    end
    return host
end
