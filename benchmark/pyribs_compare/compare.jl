# MAP-Elites on 20-dimensional Rastrigin with QualityDiversity.jl.
# Run:  julia --threads=1 benchmark/pyribs_compare/compare.jl
# The matching pyribs script is compare.py; both print the same columns.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using QualityDiversity
using Random: Xoshiro
using Statistics: median

const DIM = 20                  # solution dimension
const DIMS = (100, 100)         # archive cells
const SIGMA = 0.5               # Gaussian mutation standard deviation
const BUDGET = 360_000          # evaluations, a multiple of every batch size
const BATCH_SIZES = (36, 100, 1000)
const REPEATS = 5

const prob = Rastrigin(DIM)

# Evaluating a fixed solution lets compare.py confirm that both sides score
# the same problem.
reference = QualityDiversity.evaluate(prob, collect(range(-6, 6; length=DIM)))
println("reference fitness ", reference.fitness, " descriptor ", Tuple(reference.descriptor))

function run_once(batch_size, seed)
    alg = MAPElites(prob; dims=DIMS, batch_size, mutation=GaussianMutation(SIGMA))
    seconds = @elapsed result = solve(prob, alg; budget=BUDGET, rng=Xoshiro(seed))
    last = result.history[end]
    return (; seconds, coverage=last.coverage, max_fitness=last.max_fitness, qd_score=last.qd_score)
end

println("\nJulia $VERSION, $(Threads.nthreads()) thread(s)")
println("| batch size | min s | median s | evals/s (median) | coverage | max fitness | QD score |")
println("|---:|---:|---:|---:|---:|---:|---:|")
for batch_size in BATCH_SIZES
    run_once(batch_size, 0)                      # compile
    runs = [run_once(batch_size, seed) for seed in 1:REPEATS]
    times = [r.seconds for r in runs]
    println("| ", batch_size,
            " | ", round(minimum(times); digits=3),
            " | ", round(median(times); digits=3),
            " | ", round(Int, BUDGET / median(times)),
            " | ", round(median(r.coverage for r in runs); digits=3),
            " | ", round(median(r.max_fitness for r in runs); digits=1),
            " | ", round(Int, median(r.qd_score for r in runs)), " |")
end
