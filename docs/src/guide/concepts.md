```@meta
CurrentModule = QualityDiversity
```

# Concepts

Quality-diversity (QD) optimization looks for many good solutions that differ from
each other in ways the user cares about, not for one best solution. Each solution has
a **fitness**, how good it is, and a **descriptor**, a few numbers (or labels) that
say what kind of solution it is. An **archive** divides descriptor space into regions
and keeps the best solution found in each, its **elite**. A run fills the archive.

This page describes how the package is built from five roles, how a run proceeds, and
how to read its results.

## The five roles

| Role | Abstract type | Its job | Examples |
|---|---|---|---|
| Problem | [`AbstractQDProblem`](@ref) | score and describe a solution, draw a random one | [`QDProblem`](@ref), [`Rastrigin`](@ref QualityDiversity.Benchmarks.Rastrigin) |
| Archive | [`AbstractArchive`](@ref) | decide which solutions to keep | [`GridArchive`](@ref), [`CVTArchive`](@ref) |
| Variation | [`AbstractVariation`](@ref) | make a child from one or two parents | [`GaussianMutation`](@ref) |
| Emitter | [`AbstractEmitter`](@ref) | propose a batch of candidates each round | [`VariationEmitter`](@ref), [`CMAEmitter`](@ref) |
| Algorithm | [`QDAlgorithm`](@ref) | an archive and a tuple of emitters | [`MAPElites`](@ref), [`CMAME`](@ref) |

Each abstract type documents a short list of required and optional functions, its
*contract*. All generic code is written against these contracts, so any component that
meets its contract works with all the others. See [Extending the package](extending.md).

The interface functions, such as `evaluate`, `add!` and `vary`, are public but not
exported, because common names like `evaluate` clash with other packages. Call and
extend them with the module name:

```julia
QualityDiversity.evaluate(prob, x)
QualityDiversity.evaluate(p::MyProblem, x) = ...
```

Constructors, the solving functions and the metrics are exported.

```@docs
QualityDiversity
```

## Problems

The quickest way to define a problem is to wrap a function that returns the fitness
and the descriptor:

```@example concepts
using QualityDiversity
using Random: Xoshiro

f(x) = (-sum(abs2, x), clamp.((x[1], x[2]), -2, 2))      # fitness, descriptor
prob = QDProblem(f; init=rng -> randn(rng, 5), bounds=((-2, 2), (-2, 2)))
QualityDiversity.evaluate(prob, [0.5, -1.0, 0.0, 0.0, 0.0])
```

- **Fitness is maximized.** To minimize a cost, return its negation.
- `init` draws one random solution from the generator it is given; the archive is
  filled with such solutions before the first round.
- `bounds` gives the range of each descriptor dimension, which grid-like archives
  divide into cells. A problem whose descriptors take discrete values gives
  `categories` instead, such as `(0:5, [:small, :large])`, and gets a
  [`CategoricalArchive`](@ref).
- The function may also return a third value, the **phenotype**: the object that the
  genotype produces, when it differs from the genotype. Archives store it next to the
  solution when they are built with `phenotype=` its type.

A problem with randomness in its evaluation is a [`StochasticQDProblem`](@ref); see the
[noisy problems tutorial](../tutorials/noisy.md). A problem that needs its own type, for
example to evaluate a whole batch at once, subtypes [`AbstractQDProblem`](@ref).

## A run

An algorithm is an archive and a tuple of emitters. One round of a run is the same for
every algorithm:

```julia
for emitter in emitters
    xs  = ask(rng, emitter, archive)         # a batch of candidates
    evs = evaluate_batch(rng, problem, xs)   # their fitness and descriptors
    rs  = add_batch!(archive, xs, evs)       # what the archive did with each
    tell!(emitter, xs, evs, rs)              # the emitter learns from it
end
```

Different algorithms are different archives and emitters, not different loops.
The functions of [CommonSolve.jl](https://github.com/SciML/CommonSolve.jl) run it:

```@example concepts
alg = MAPElites(prob; dims=(10, 10), batch_size=50)
result = solve(prob, alg; budget=20_000, rng=Xoshiro(1))
```

`solve` is [`init`](@ref) followed by [`solve!`](@ref). To look at the state between
rounds, or to stop on a condition of your own, step by hand:

```@example concepts
state = init(prob, alg; budget=20_000, rng=Xoshiro(1))
while state.n_evals < 20_000 && coverage(state.archive) < 0.9
    step!(state)
end
state.n_evals, coverage(state.archive)
```

`init` copies the archive and emitters of `alg`, so the same algorithm can be solved
several times. The `budget` counts evaluations; a run stops at the end of the round
that reaches it. `callback(state)` is called after initialization and after every
round.

## Results

[`solve`](@ref) returns a [`QDResult`](@ref QualityDiversity.QDResult) with three fields:

- `archive`: the archive of the best solutions found;
- `history`: the metrics after initialization and after every round;
- `n_evals`: the number of evaluations spent.

```@example concepts
result.history[end]
```

[`elites`](@ref QualityDiversity.elites) iterates over the stored solutions and their
evaluations:

```@example concepts
best_x, best_ev = argmax(e -> e[2].fitness, collect(QualityDiversity.elites(result.archive)))
best_ev
```

### Metrics

| Metric | Meaning |
|---|---|
| [`coverage`](@ref) | fraction of the cells that hold a solution |
| [`max_fitness`](@ref) | fitness of the best elite |
| [`mean_fitness`](@ref) | mean fitness of the elites |
| [`qd_score`](@ref) | sum of the fitness of the elites, each raised by `offset` |
| [`cqd_score`](@ref) | continuous QD score, which does not depend on the cells |

The QD score sums fitness over the cells, so a negative fitness would lower it when a
cell is filled. `qd_score(archive; offset)` adds `offset` to every elite; it should
make every attainable fitness non-negative. A problem can supply its offset with
[`fitness_offset`](@ref QualityDiversity.fitness_offset), which the history uses.

Archives with several objectives have metrics of their own, such as the MOQD score;
see [`ParetoGridArchive`](@ref).

## Randomness and threads

Every function that draws random numbers takes the generator as its first argument,
and a run takes its generator as `rng`. A run with a seeded generator is reproducible:

```@example concepts
a = solve(prob, alg; budget=5_000, rng=Xoshiro(7))
b = solve(prob, alg; budget=5_000, rng=Xoshiro(7))
a.history == b.history
```

The default `evaluate_batch` evaluates the solutions of a batch on several threads
(start Julia with `--threads=auto` to use them), so `evaluate` must be thread-safe. The
results do not depend on the number of threads: a stochastic problem draws one seed
per solution from the run's generator before the threaded section, in the order of
the batch. Evaluation is the only part that runs on several threads; the archive and
the emitters run on one.
