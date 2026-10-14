# CMA-ME and CMA-MAE on 20-dimensional Rastrigin with QualityDiversity.jl.
# Run:  julia --threads=1 benchmark/compare/compare_cma.jl
# The matching scripts are compare_cma.py (pyribs) and compare_cma_qdax.py (QDax);
# all print the same columns.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
VERSION < v"1.11" && Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using BenchmarkTools
using QualityDiversity
using Random: Xoshiro
using Statistics: median

const DIM = 20                  # solution dimension
const DIMS = (100, 100)         # archive cells
const SIGMA0 = 1.0              # initial step size of every search
const EMITTERS = 5
const BATCH = 12                # 4 + floor(3 log(DIM)), the CMA-ES default
const BUDGET = 100_020          # evaluations, a multiple of EMITTERS * BATCH and of BATCH
const REPEATS = parse(Int, get(ENV, "REPEATS", "11"))   # timed runs per method, one seed each
const OFFSET = 41.0 * DIM       # the same QD-score offset as in the other scripts
const THRESHOLD_MIN = -OFFSET   # lowest fitness of Rastrigin, for CMA-MAE
const LEARNING_RATE = 0.01

const prob = Rastrigin(DIM)
const x0 = zeros(DIM)

# Each method is a constructor; all start every search at the origin, as the
# other two libraries do for their first search.
const METHODS = (
    "CMA-ME, improvement emitters" =>
        () -> CMAME(prob; dims=DIMS, sigma0=SIGMA0, num_emitters=EMITTERS, batch_size=BATCH, x0),
    "CMA-ME, optimizing emitters" =>
        () -> CMAME(prob; dims=DIMS, sigma0=SIGMA0, num_emitters=EMITTERS, batch_size=BATCH, x0,
                    ranker=ObjectiveRanker(), selection=MuSelection(), restart=ConvergenceRestart()),
    "CMA-ME, random direction emitters" =>
        () -> CMAME(prob; dims=DIMS, sigma0=SIGMA0, num_emitters=EMITTERS, batch_size=BATCH, x0,
                    ranker=TwoStageRandomDirectionRanker()),
    "CMA-MAE" =>
        () -> CMAMAE(prob; dims=DIMS, sigma0=SIGMA0, num_emitters=EMITTERS, batch_size=BATCH, x0,
                     threshold_min=THRESHOLD_MIN, learning_rate=LEARNING_RATE),
)

const seed = Ref(0)
const finals = []

function benchmark_method(make)
    empty!(finals)
    seed[] = 0
    trial = @benchmark begin
        result = solve(prob, alg; budget=BUDGET, rng=Xoshiro(seed[]))
        push!(finals, (coverage=coverage(result.archive),
                       max_fitness=Float64(max_fitness(result.archive)),
                       qd_score=qd_score(result.archive; offset=OFFSET)))
    end setup = (alg = $make(); seed[] += 1) evals = 1 samples = REPEATS seconds = 900
    return trial, copy(finals)
end

println("Julia $VERSION, $(Threads.nthreads()) thread(s), $REPEATS runs per row, budget $BUDGET")
println("| method | min s | median s | evals/s (median) | coverage | max fitness | QD score |")
println("|---|---:|---:|---:|---:|---:|---:|")
for (name, make) in METHODS
    benchmark_method(make)                        # compile
    trial, runs = benchmark_method(make)
    times = trial.times ./ 1e9
    println("| ", name,
            " | ", round(minimum(times); digits=3),
            " | ", round(median(times); digits=3),
            " | ", round(Int, BUDGET / median(times)),
            " | ", round(median(r.coverage for r in runs); digits=3),
            " | ", round(median(r.max_fitness for r in runs); digits=1),
            " | ", round(Int, median(r.qd_score for r in runs)), " |")
end
