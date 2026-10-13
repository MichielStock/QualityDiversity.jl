# MAP-Elites on 20-dimensional Rastrigin with QualityDiversity.jl.
# Run:  julia --threads=1 benchmark/compare/compare.jl
# The matching pyribs script is compare.py; both print the same columns.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
VERSION < v"1.11" && Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using BenchmarkTools
using QualityDiversity
using Random: Xoshiro
using Statistics: median, quantile

const DIM = 20                  # solution dimension
const DIMS = (100, 100)         # archive cells
const SIGMA = 0.5               # Gaussian mutation standard deviation
const BUDGET = 360_000          # evaluations, a multiple of every batch size
const BATCH_SIZES = (36, 100, 1000, 10_000)
const REPEATS = 21              # timed runs per batch size, one seed each

const OFFSET = 41.0 * DIM       # the same QD-score offset as Rastrigin in Python

# The Float64 problem is the Rastrigin benchmark of the package.
const prob = Rastrigin(DIM)

# The Float32 problem is the same function written with Float32 arithmetic, the
# way a user would define it with QDProblem. The archive then holds
# Vector{Float32} genotypes and Float32 fitness values, and the mutation
# produces Float32 children.
function rastrigin32(x::Vector{Float32})
    limit = 5.12f0
    half = DIM ÷ 2
    total = 0f0
    first_half = 0f0
    second_half = 0f0
    for i in 1:DIM
        xi = clamp(x[i], -limit, limit)
        total += xi^2 - 10f0 * cos(2f0 * Float32(π) * xi)
        if i <= half
            first_half += xi
        else
            second_half += xi
        end
    end
    return -(10f0 * DIM + total),
           (clamp(first_half / half, -limit, limit), clamp(second_half / (DIM - half), -limit, limit))
end

const prob32 = QDProblem(rastrigin32;
                         init=rng -> (2f0 .* rand(rng, Float32, DIM) .- 1f0) .* 5.12f0,
                         bounds=((-5.12, 5.12), (-5.12, 5.12)))

make_algorithm(::Type{Float64}, batch_size) =
    MAPElites(prob; dims=DIMS, batch_size, mutation=GaussianMutation(SIGMA))
make_algorithm(::Type{Float32}, batch_size) =
    MAPElites(prob32; dims=DIMS, batch_size, mutation=GaussianMutation(Float32(SIGMA)),
              genotype=Vector{Float32}, fitness=Float32)
problem(::Type{Float64}) = prob
problem(::Type{Float32}) = prob32

# Evaluating a fixed solution lets compare.py confirm that both sides score
# the same problem.
reference = QualityDiversity.evaluate(prob, collect(range(-6, 6; length=DIM)))
println("reference fitness ", reference.fitness, " descriptor ", Tuple(reference.descriptor))

const seed = Ref(0)
const finals = []           # final statistics of every timed run

function benchmark_batch(T, batch_size)
    empty!(finals)
    seed[] = 0
    p = problem(T)
    # One timed run per sample (`evals=1`), each with a fresh algorithm and seed.
    trial = @benchmark begin
        result = solve($p, alg; budget=BUDGET, rng=Xoshiro(seed[]))
        push!(finals, (coverage=coverage(result.archive),
                       max_fitness=Float64(max_fitness(result.archive)),
                       qd_score=qd_score(result.archive; offset=OFFSET)))
    end setup = (alg = make_algorithm($T, $batch_size); seed[] += 1) evals = 1 samples = REPEATS seconds = 600
    return trial, copy(finals)
end

for T in (Float64, Float32)
    println("\nJulia $VERSION, $(Threads.nthreads()) thread(s), $T genotypes, $REPEATS runs per row")
    println("| batch size | min s | median s | middle half s | evals/s (median) | coverage | max fitness | QD score |")
    println("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for batch_size in BATCH_SIZES
        benchmark_batch(T, batch_size)               # compile
        trial, runs = benchmark_batch(T, batch_size)
        times = trial.times ./ 1e9
        lo, hi = quantile(times, (0.25, 0.75))
        println("| ", batch_size,
                " | ", round(minimum(times); digits=3),
                " | ", round(median(times); digits=3),
                " | ", round(lo; digits=3), "–", round(hi; digits=3),
                " | ", round(Int, BUDGET / median(times)),
                " | ", round(median(r.coverage for r in runs); digits=3),
                " | ", round(median(r.max_fitness for r in runs); digits=1),
                " | ", round(Int, median(r.qd_score for r in runs)), " |")
    end
end
