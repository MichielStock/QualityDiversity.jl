# MAP-Elites on 20-dimensional Rastrigin with QualityDiversity.jl.
# Run:  julia --threads=1 benchmark/pyribs_compare/compare.jl
# The matching pyribs script is compare.py; both print the same columns.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using BenchmarkTools
using QualityDiversity
using Random: Xoshiro
using Statistics: median, quantile

const DIM = 20                  # solution dimension
const DIMS = (100, 100)         # archive cells
const SIGMA = 0.5               # Gaussian mutation standard deviation
const BUDGET = 360_000          # evaluations, a multiple of every batch size
const BATCH_SIZES = (36, 100, 1000)
const REPEATS = 21              # timed runs per batch size, one seed each

const prob = Rastrigin(DIM)

# Evaluating a fixed solution lets compare.py confirm that both sides score
# the same problem.
reference = QualityDiversity.evaluate(prob, collect(range(-6, 6; length=DIM)))
println("reference fitness ", reference.fitness, " descriptor ", Tuple(reference.descriptor))

const seed = Ref(0)
const finals = []           # last history entry of every timed run

function benchmark_batch(batch_size)
    empty!(finals)
    seed[] = 0
    # One timed run per sample (`evals=1`), each with a fresh algorithm and seed.
    trial = @benchmark begin
        result = solve(prob, alg; budget=BUDGET, rng=Xoshiro(seed[]))
        push!(finals, result.history[end])
    end setup = (alg = MAPElites(prob; dims=DIMS, batch_size=$batch_size,
                                 mutation=GaussianMutation(SIGMA));
                 seed[] += 1) evals = 1 samples = REPEATS seconds = 600
    return trial, copy(finals)
end

println("\nJulia $VERSION, $(Threads.nthreads()) thread(s), $REPEATS runs per row")
println("| batch size | min s | median s | middle half s | evals/s (median) | coverage | max fitness | QD score |")
println("|---:|---:|---:|---:|---:|---:|---:|---:|")
for batch_size in BATCH_SIZES
    benchmark_batch(batch_size)                  # compile
    trial, runs = benchmark_batch(batch_size)
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
