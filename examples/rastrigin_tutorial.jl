# # MAP-Elites on the Rastrigin function
#
# Run it from the repository root:
#
#     julia --project examples/rastrigin_tutorial.jl
#
# Quality-diversity (QD) optimization does not look for one best solution. It
# fills a grid of "niches", each defined by a descriptor, with the best
# solution found for that niche.

using QualityDiversity
using Random: Xoshiro

# ## 1. The problem
#
# `Rastrigin(10)` is a benchmark over 10 real parameters. A solution's fitness
# is the negated Rastrigin value (higher is better, maximum 0). Its descriptor
# is two numbers: the mean of the first five parameters and the mean of the
# last five, each in [-5.12, 5.12].

prob = Rastrigin(10)

x = QualityDiversity.random_solution(Xoshiro(1), prob)
ev = QualityDiversity.evaluate(prob, x)
println("one random solution: fitness ", round(ev.fitness; digits=1),
        ", descriptor ", round.(ev.descriptor; digits=2))

# ## 2. The algorithm
#
# MAP-Elites keeps a grid over the descriptor space. Each round it picks
# `batch_size` solutions from the grid, mutates them with Gaussian noise of
# size `σ`, and keeps a child if it beats the current occupant of its cell.

alg = MAPElites(prob; dims=(20, 20), batch_size=100, variation=GaussianMutation(0.3))

# ## 3. Run it
#
# The budget counts evaluations. Passing a seeded `rng` makes the run
# reproducible.

result = solve(prob, alg; budget=100_000, rng=Xoshiro(1))

# ## 4. Read the results
#
# `result.history` has one entry per round with the archive metrics.

for record in result.history[[1, 100, end]]
    println("after ", lpad(record.n_evals, 6), " evaluations: coverage ",
            round(record.coverage; digits=3), ", max fitness ",
            round(record.max_fitness; digits=1), ", QD score ",
            round(record.qd_score; digits=0))
end

# The archive itself is in `result.archive`. The metric functions work on it
# directly, and `QualityDiversity.elites` iterates over the stored solutions.

archive = result.archive
println("\n", length(archive), " of ", QualityDiversity.capacity(archive),
        " cells filled (coverage ", round(coverage(archive); digits=3), ")")
println("best fitness: ", round(max_fitness(archive); digits=2))

best_x, best_ev = argmax(pair -> pair[2].fitness, collect(QualityDiversity.elites(archive)))
println("best solution: ", round.(best_x; digits=2))

# A text picture of the grid: darker means higher fitness, blank means empty.
# Rows are the second descriptor, columns the first.

shades = collect(" ░▒▓█")
fitnesses = fill(NaN, 20, 20)
for (_, ev) in QualityDiversity.elites(archive)
    i = QualityDiversity.cell_index(archive, ev.descriptor) - 1
    fitnesses[i ÷ 20 + 1, i % 20 + 1] = ev.fitness
end
lo, hi = extrema(filter(!isnan, fitnesses))
println()
for row in 1:20
    println(join(isnan(f) ? ' ' : shades[2 + round(Int, 3 * (f - lo) / (hi - lo))]
                 for f in fitnesses[row, :]))
end

#md # With a Makie backend loaded, the archive draws as a heatmap; see
#md # [Plotting archives](heatmap.md).
#md
#md using CairoMakie
#md heatmap(archive; axis=(xlabel="descriptor 1", ylabel="descriptor 2", aspect=1))

# ## 5. Things to try
#
# - A different mutation size: `GaussianMutation(0.1)` or `GaussianMutation(1.0)`.
# - A crossover that moves along the line between two elites:
#   `variation=IsoLineVariation(0.01, 0.2)`.
# - A mutation that keeps every entry inside the box [-5.12, 5.12]:
#   `variation=PolynomialMutation(-5.12, 5.12; eta=20)`. With Gaussian mutation
#   most stored solutions drift outside it, and `Rastrigin` clips them when scoring.
# - A finer grid: `dims=(40, 40)`.
# - Zoom into part of the descriptor space and ignore everything outside:
#   `MAPElites(prob; dims=(20, 20), bounds=((-2, 2), (-2, 2)), outside=RejectOutside())`.
# - Step one round at a time instead of calling `solve`:
#   `state = init(prob, alg; budget=10_000, rng=Xoshiro(1)); step!(state)`.
