# MAP-Elites on 20-dimensional Rastrigin with QualityDiversity.jl on a CUDA GPU: the
# archive, the variation and the evaluation of every batch run on the device, with a
# DeviceGridArchive and with a DeviceCVTArchive whose centroids are the centers of the
# grid cells, the centroids that compare_qdax.py gives QDax's centroid search.
# Run:  julia benchmark/compare/gpu/compare_gpu.jl
# On a machine with several GPUs, choose one with CUDA_DEVICE_ORDER=PCI_BUS_ID and
# CUDA_VISIBLE_DEVICES=<index in nvidia-smi>; without CUDA_DEVICE_ORDER the CUDA runtime
# numbers the devices fastest first.
# The problem, grid, mutation, batch sizes, budget and columns are those of
# ../compare.jl; compare_qdax.py runs QDax on the same device.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using BenchmarkTools
using CUDA
using QualityDiversity
using Random: Xoshiro
using Statistics: median, quantile

const DIM = 20
const DIMS = (100, 100)
const SIGMA = 0.5
const BUDGET = 360_000
const BATCH_SIZES = (36, 100, 1000, 10_000)
const REPEATS = 21
const LIMIT = 5.12
const OFFSET = 41.0 * DIM

# The Rastrigin benchmark written for any element type, so that the kernel computes in
# the precision of the archive. A device kernel runs it on one solution per thread.
function rastrigin(x::AbstractVector{T}) where {T}
    limit = T(LIMIT)
    half = length(x) ÷ 2
    total = zero(T)
    first_half = zero(T)
    second_half = zero(T)
    for i in eachindex(x)
        xi = clamp(x[i], -limit, limit)
        total += xi^2 - 10 * cospi(2 * xi)
        i - firstindex(x) < half ? (first_half += xi) : (second_half += xi)
    end
    return -(10 * T(length(x)) + total),
           (clamp(first_half / half, -limit, limit), clamp(second_half / (length(x) - half), -limit, limit))
end

problem(::Type{T}) where {T} = QDProblem(rastrigin; init=rng -> (2 .* rand(rng, T, DIM) .- 1) .* T(LIMIT),
                                         bounds=((-LIMIT, LIMIT), (-LIMIT, LIMIT)))

# The centers of the cells of the grid, in the order of QDax's compute_euclidean_centroids.
const CENTERS = vec([[-LIMIT + (i - 0.5) * 2LIMIT / DIMS[1], -LIMIT + (j - 0.5) * 2LIMIT / DIMS[2]]
                     for i in 1:DIMS[1], j in 1:DIMS[2]])

make_algorithm(::Val{:grid}, T, batch_size) =
    MAPElites(problem(T); dims=DIMS, batch_size, variation=GaussianMutation(T(SIGMA)), device=CuArray,
              genotype=Vector{T}, fitness=T)
make_algorithm(::Val{:cvt}, T, batch_size) =
    MAPElites(DeviceCVTArchive{T,T}(CuArray, CENTERS, ((-LIMIT, LIMIT), (-LIMIT, LIMIT)), DIM);
              batch_size, variation=GaussianMutation(T(SIGMA)))

const seed = Ref(0)
const finals = []

function benchmark_batch(archive, T, batch_size)
    empty!(finals)
    seed[] = 0
    p = problem(T)
    # `solve` returns once the last batch is on the device; the metrics recorded after
    # every step wait for it, so the time covers the whole run.
    trial = @benchmark begin
        result = solve($p, alg; budget=BUDGET, rng=Xoshiro(seed[]))
        push!(finals, (coverage=coverage(result.archive),
                       max_fitness=Float64(max_fitness(result.archive)),
                       qd_score=qd_score(result.archive; offset=OFFSET)))
    end setup = (alg = make_algorithm($archive, $T, $batch_size); seed[] += 1) evals = 1 samples = REPEATS seconds = 900
    return trial, copy(finals)
end

for (name, archive) in (("DeviceGridArchive", Val(:grid)), ("DeviceCVTArchive", Val(:cvt))), T in (Float32, Float64)
    println("\nJulia $VERSION, CUDA ", CUDA.name(CUDA.device()), ", $name, $T, $REPEATS runs per row")
    println("| batch size | min s | median s | middle half s | evals/s (median) | coverage | max fitness | QD score |")
    println("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for batch_size in BATCH_SIZES
        benchmark_batch(archive, T, batch_size)               # compile
        trial, runs = benchmark_batch(archive, T, batch_size)
        times = trial.times ./ 1e9
        lo, hi = quantile(times, (0.25, 0.75))
        println("| ", batch_size,
                " | ", round(minimum(times); digits=4),
                " | ", round(median(times); digits=4),
                " | ", round(lo; digits=4), "–", round(hi; digits=4),
                " | ", round(Int, BUDGET / median(times)),
                " | ", round(median(r.coverage for r in runs); digits=3),
                " | ", round(median(r.max_fitness for r in runs); digits=1),
                " | ", round(Int, median(r.qd_score for r in runs)), " |")
    end
end
