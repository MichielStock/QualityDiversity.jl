# BenchmarkTools suite for the hot path. Run it with `julia benchmark/run.jl`.
# Leave cores free for the machine while it runs; the threaded evaluate_batch
# uses the threads Julia was started with.
using BenchmarkTools
using QualityDiversity
using Random: Xoshiro
using StaticArrays: SVector

const QD = QualityDiversity
const SUITE = BenchmarkGroup()

const DIM = 20
const prob = Rastrigin(DIM)
const rng = Xoshiro(1)
const x = QD.random_solution(rng, prob)
const ev = QD.evaluate(prob, x)
const BATCH = 100

new_archive() = GridArchive(prob; dims=(100, 100))

function filled_archive(n)
    archive = new_archive()
    xs = [QD.random_solution(rng, prob) for _ in 1:n]
    QD.add_batch!(archive, xs, QD.evaluate_batch(prob, xs))
    return archive
end

# ── problem ───────────────────────────────────────────────
SUITE["problem"]["evaluate"] = @benchmarkable QD.evaluate($prob, $x)
for n in (36, 100, 1000)
    xs = [QD.random_solution(rng, prob) for _ in 1:n]
    SUITE["problem"]["evaluate_batch"]["batch=$n"] = @benchmarkable QD.evaluate_batch($prob, $xs)
end

# ── variation ─────────────────────────────────────────────
const mutation = GaussianMutation(0.5)
SUITE["variation"]["GaussianMutation"] = @benchmarkable QD.vary($rng, $mutation, $x)
const isoline = IsoLineVariation(0.01, 0.2)
const x2 = QD.random_solution(rng, prob)
SUITE["variation"]["IsoLineVariation"] = @benchmarkable QD.vary($rng, $isoline, $x, $x2)
const polynomial = PolynomialMutation(-5.12, 5.12; eta=20)
SUITE["variation"]["PolynomialMutation"] = @benchmarkable QD.vary($rng, $polynomial, $x)
const uniform = UniformCrossover()
SUITE["variation"]["UniformCrossover"] = @benchmarkable QD.vary($rng, $uniform, $x, $x2)

# ── archive ───────────────────────────────────────────────
const archive = filled_archive(20_000)
const d = ev.descriptor
SUITE["archive"]["cell_index"] = @benchmarkable QD.cell_index($archive, $d)
SUITE["archive"]["add!/rejected"] = @benchmarkable QD.add!($archive, $x, $ev)
SUITE["archive"]["add!/new cell"] = @benchmarkable QD.add!(a, $x, $ev) setup = (a = new_archive()) evals = 1
SUITE["archive"]["add!/improved"] =
    @benchmarkable QD.add!(a, $x, $(QD.Evaluation(ev.fitness + 1, ev.descriptor))) setup =
        (a = new_archive(); QD.add!(a, $x, $ev)) evals = 1
SUITE["archive"]["sample_elites/n=100"] = @benchmarkable QD.sample_elites($rng, $archive, 100)
SUITE["archive"]["add_batch!/n=100"] = let xs = [QD.random_solution(rng, prob) for _ in 1:BATCH]
    evs = QD.evaluate_batch(prob, xs)
    @benchmarkable QD.add_batch!($archive, $xs, $evs)
end
SUITE["archive"]["metrics"]["coverage"] = @benchmarkable coverage($archive)
SUITE["archive"]["metrics"]["qd_score"] = @benchmarkable qd_score($archive; offset=800.0)
SUITE["archive"]["metrics"]["max_fitness"] = @benchmarkable max_fitness($archive)
SUITE["archive"]["metrics"]["mean_fitness"] = @benchmarkable mean_fitness($archive)

const cvt = let a = CVTArchive(Xoshiro(3), prob; cells=1000, samples=20_000)
    xs = [QD.random_solution(rng, prob) for _ in 1:20_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(prob, xs))
    a
end
SUITE["archive"]["CVTArchive/cell_index (1000 cells)"] = @benchmarkable QD.cell_index($cvt, $d)
SUITE["archive"]["CVTArchive/add!/rejected"] = @benchmarkable QD.add!($cvt, $x, $ev)
SUITE["archive"]["CVTArchive/sample_elites/n=100"] = @benchmarkable QD.sample_elites($rng, $cvt, 100)

# ── emitter ───────────────────────────────────────────────
const emitter = VariationEmitter(mutation; batch_size=BATCH)
SUITE["emitter"]["ask/batch=100"] = @benchmarkable QD.ask($rng, $emitter, $archive)

# ── algorithm ─────────────────────────────────────────────
const alg = MAPElites(prob; dims=(100, 100), batch_size=BATCH, variation=mutation)
const state = init(prob, alg; budget=typemax(Int), rng=Xoshiro(2))
for _ in 1:200
    step!(state)    # reach a steady state before measuring
end
SUITE["algorithm"]["step!/batch=100"] = @benchmarkable step!($state)
SUITE["algorithm"]["solve/360k evaluations"] =
    @benchmarkable solve($prob, $alg; budget=360_000, rng=Xoshiro(3)) samples = 5 evals = 1 seconds = 120

const unstructured = let a = UnstructuredArchive(prob; capacity=1000, threshold=0.2)
    xs = [QD.random_solution(rng, prob) for _ in 1:20_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(prob, xs))
    a
end
SUITE["archive"]["UnstructuredArchive/cell_index (1000 slots)"] = @benchmarkable QD.cell_index($unstructured, $d)
SUITE["archive"]["UnstructuredArchive/add!/rejected"] = @benchmarkable QD.add!($unstructured, $x, $ev)
SUITE["archive"]["UnstructuredArchive/sample_elites/n=100"] = @benchmarkable QD.sample_elites($rng, $unstructured, 100)
