# # MAP-Elites on the Rastrigin function
#
# Quality-diversity (QD) optimization does not look for one best solution. It
# fills a grid of cells, each covering a region of descriptor space, with the
# best solution found for that cell. This tutorial goes through the workflow of
# the package once:
#
# 1. define a problem,
# 2. choose an algorithm,
# 3. run it with `solve`, and
# 4. look at the result: metrics, elites and a heatmap.
#
# Set up once, from the repository root:
#
#     julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run:
#
#     julia --project=examples examples/rastrigin_tutorial.jl

using QualityDiversity
using Random: Xoshiro

# ## 1. The problem
#
# `Rastrigin(10)` is a benchmark over 10 real parameters. The fitness of a
# solution is the negated Rastrigin value, so higher is better, with a maximum of
# 0. Its descriptor is two numbers: the mean of the first five parameters and the
# mean of the last five, each in [-5.12, 5.12].

prob = Rastrigin(10)

# For your own problem, `QDProblem(f; init, bounds)` wraps a function `f` that
# returns `(fitness, descriptor)`; see the [graph tutorial](graphs.md).

# ## 2. The algorithm
#
# MAP-Elites keeps a grid over descriptor space, here 20 by 20 cells. Each round
# it draws `batch_size` parents from the grid, mutates them with Gaussian noise of
# standard deviation 0.3, and a child takes a cell if it is empty or if the child
# beats the solution there.

alg = MAPElites(prob; dims=(20, 20), batch_size=100, variation=GaussianMutation(0.3))
nothing #hide

# ## 3. Running it
#
# The budget counts evaluations. A seeded random number generator makes the run
# reproducible.

result = solve(prob, alg; budget=100_000, rng=Xoshiro(1))

# ## 4. The result
#
# `solve` returns a summary of the final archive. `result.history` holds the same
# metrics after every round, so you can follow the run, from the first round to
# the last:

result.history[1]

#-

result.history[end]

# `result.archive` is the grid. Its elites, the best solution of every filled
# cell, come as pairs of a solution and its evaluation:

archive = result.archive
elites = collect(QualityDiversity.elites(archive))
length(elites)

#-

x, ev = first(elites)
(fitness=ev.fitness, descriptor=ev.descriptor)

# The best elite of the whole archive:

best_x, best_ev = argmax(((x, ev),) -> ev.fitness, elites)
round.(best_x; digits=2)

# With a Makie backend loaded, such as CairoMakie, `heatmap` draws the fitness of
# every cell; empty cells stay blank. [Plotting archives](heatmap.md) shows more.

using CairoMakie

fig, ax, hm = heatmap(archive; figure=(size=(560, 460),),
                      axis=(xlabel="descriptor 1", ylabel="descriptor 2", aspect=1))
Colorbar(fig[1, 2], hm; label="fitness")
save(joinpath(@__DIR__, "rastrigin.png"), fig) #src
fig #md

# ## Things to try
#
# - A different mutation size: `GaussianMutation(0.1)` or `GaussianMutation(1.0)`.
# - A crossover that moves along the line between two elites:
#   `variation=IsoLineVariation(0.01, 0.2)`.
# - A finer grid: `dims=(40, 40)`.
# - Stepping one round at a time instead of calling `solve`:
#   `state = init(prob, alg; budget=10_000, rng=Xoshiro(1)); step!(state)`.
# - Another algorithm, such as `CMAME(prob; dims=(20, 20), sigma0=0.5)`: the rest
#   of the workflow stays the same.
