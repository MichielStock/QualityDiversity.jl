# MOME on the two-objective BiSphere benchmark with QualityDiversity.jl.
# Run:  julia --threads=1 benchmark/compare/compare_mome.jl
# The matching script for QDax is compare_mome_qdax.py; both print the same columns.
# pyribs has no multi-objective archive.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
VERSION < v"1.11" && Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using BenchmarkTools
using QualityDiversity
using Random: Xoshiro
using Statistics: median

const DIM = 10                  # solution dimension
const DIMS = (20, 20)           # archive cells
const SIGMA = 0.2               # Gaussian mutation standard deviation
const BATCH = 64
const MAX_FRONT = 50            # solutions kept in the Pareto front of a cell
const BUDGET = 100_032          # evaluations, a multiple of the batch size
const REPEATS = parse(Int, get(ENV, "REPEATS", "11"))   # timed runs, one seed each

const prob = BiSphere(DIM)
const REFERENCE = QualityDiversity.reference_point(prob)

# Fitness and descriptor of one fixed solution, to compare with compare_mome_qdax.py.
reference = QualityDiversity.evaluate(prob, collect(range(-6, 6; length=DIM)))
println("reference fitness ", Tuple(reference.fitness), " descriptor ", Tuple(reference.descriptor),
        " reference point ", Tuple(REFERENCE))

make_algorithm() = MOME(prob; dims=DIMS, max_front=MAX_FRONT, batch_size=BATCH,
                        variation=GaussianMutation(SIGMA))

const seed = Ref(0)
const finals = []

function benchmark_mome()
    empty!(finals)
    seed[] = 0
    trial = @benchmark begin
        result = solve(prob, alg; budget=BUDGET, rng=Xoshiro(seed[]))
        a = result.archive
        push!(finals, (coverage=coverage(a), moqd_score=qd_score(a), global_hypervolume=global_hypervolume(a),
                       max_hypervolume=max_hypervolume(a), solutions=num_solutions(a)))
    end setup = (alg = make_algorithm(); seed[] += 1) evals = 1 samples = REPEATS seconds = 900
    return trial, copy(finals)
end

println("\nJulia $VERSION, $(Threads.nthreads()) thread(s), $REPEATS runs, budget $BUDGET")
println("| min s | median s | evals/s (median) | coverage | MOQD score | global hypervolume | max hypervolume | solutions |")
println("|---:|---:|---:|---:|---:|---:|---:|---:|")
benchmark_mome()                                   # compile
trial, runs = benchmark_mome()
times = trial.times ./ 1e9
println("| ", round(minimum(times); digits=3),
        " | ", round(median(times); digits=3),
        " | ", round(Int, BUDGET / median(times)),
        " | ", round(median(r.coverage for r in runs); digits=3),
        " | ", round(Int, median(r.moqd_score for r in runs)),
        " | ", round(Int, median(r.global_hypervolume for r in runs)),
        " | ", round(Int, median(r.max_hypervolume for r in runs)),
        " | ", round(Int, median(r.solutions for r in runs)), " |")
