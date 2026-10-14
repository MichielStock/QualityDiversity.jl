# QualityDiversity.jl — project guide

This file is the design reference and working agreement for this repository.
Read it fully before writing code. When a decision below conflicts with an
instinct, follow the decision; when something is undecided (see
[Open decisions](#open-decisions)), ask the author instead of choosing.

The package name is `QualityDiversity`. It is not registered in the General
registry as of 2026-10-13; confirm before registering.

---

## 1. What this package is

A Julia library for quality-diversity (QD) optimization: MAP-Elites and the
algorithms built from the same parts (CMA-ME, MOME, novelty-based archives,
differentiable QD).

**What distinguishes it from pyribs and QDax.** Both Python libraries already
offer interchangeable archives and emitters, and QDax is very fast for
problems that fit JAX: fixed-shape arrays and jittable scoring functions.
Neither handles *structured, variable-size, discrete genotypes* well —
molecules, reaction networks, polyomino assemblies, graphs, syntax trees,
recipes. In Python those become slow object loops; in JAX they need padding
or do not work at all.

This package targets that gap: **fast QD for any genotype a user can write as
a Julia type.** The user defines a type and a few methods; the library
compiles specialized code for it. Continuous vectors are fully supported and
get good defaults, but they are not the only first-class case.

Design goal: simple from the outside, fast on the inside.

- Outside: a user with a fitness function and a descriptor function gets a
  filled archive in a few lines. Switching algorithm is one argument.
- Inside: concretely typed containers, no allocation in the archive hot path,
  no runtime dispatch in the main loop.

Speed matters most when evaluation is cheap (benchmarks, analytic problems,
fast simulators). When evaluation is expensive (calling an ML model, a
chemistry toolkit), the selling points are ergonomics and composability.
Neither should be sacrificed for the other.

### Out of scope (for now)

- Reinforcement-learning environments and neural-policy training loops.
- A plotting backend in the core package (plotting lives in an extension).
- Distributed (multi-node) evaluation. Threaded and batched evaluation are in
  scope.

---

## 2. Design principles

1. **Roles are abstract types; behavior is a small set of functions.** Every
   component plays one of five roles (Problem, Archive, Variation, Emitter,
   Algorithm). Each role has an abstract type and a documented contract of
   required and optional methods. Generic code is written only against those
   contracts.
2. **One loop, many algorithms.** Most QD algorithms are the same
   ask → evaluate → add → tell loop with a different archive and emitters.
   There is one generic loop. `MAPElites(...)`, `CMAME(...)` and friends are
   convenience constructors that assemble components, not separate loops.
3. **Dispatch, not flags.** Algorithm variants are chosen by passing a
   different type, never by a `Symbol` or `Bool` switch.
4. **Type parameters carry the hot-path types.** Archives are parametrized by
   genotype `G`, fitness `F` and descriptor `D`, so everything downstream is
   concretely typed and batch outputs can be preallocated.
5. **Batch-first.** Emitters produce batches; evaluation is batched. This is
   the hook for threading, GPU, and calls out to Python.
6. **Explicit randomness.** Every function that draws random numbers takes an
   `AbstractRNG` as its first argument. A run is reproducible from its seed.
7. **Fail fast.** Invalid input throws an error with a message that tells the
   user what is wrong. Never silently clamp, skip, or continue.
8. **Interface functions are `public`, not exported.** Users extend them with
   qualified names (`QualityDiversity.evaluate(p::MyProblem, x) = ...`), as
   with `Base.show`. This avoids clashes such as `evaluate` (Distances.jl) and
   `sample` (StatsBase.jl). Constructors, the CommonSolve verbs (`init`, `step!`,
   `solve!`, `solve`), and metrics are exported.

---

## 3. Core abstractions

This is the target shape of `src/interfaces.jl` and `src/evaluation.jl`.
Signatures are normative. `GridArchive`, `VariationEmitter` and the loop
implement them; the other concrete types in the layout follow the roadmap.

```julia
# ── Results of one evaluation ─────────────────────────────
struct Evaluation{F,D}
    fitness::F        # Real (single objective) or AbstractVector (multi-objective)
    descriptor::D     # usually SVector{N,Float64}
end

@enum AddStatus Rejected Improved NewCell

struct AddResult{F}
    status::AddStatus
    improvement::F    # fitness gain over the previous occupant (or over the
                      # archive's threshold for a new cell); used by CMA-ME ranking
end

# ── Problem ───────────────────────────────────────────────
abstract type AbstractQDProblem end

# ── Archive ───────────────────────────────────────────────
abstract type AbstractArchive{G,F,D} end

# ── Variation ─────────────────────────────────────────────
abstract type AbstractVariation end

# ── Emitter ───────────────────────────────────────────────
abstract type AbstractEmitter end

# ── Algorithm, state, result ──────────────────────────────
struct QDAlgorithm{A<:AbstractArchive,E<:Tuple}
    archive::A
    emitters::E
end
```

### Contracts

Fitness is **maximized**. Users who minimize return the negated value.

**`AbstractQDProblem`**

| Function | Status | Meaning |
|---|---|---|
| `evaluate(prob, x) -> Evaluation` | required | fitness and descriptor of one solution |
| `random_solution(rng, prob)` | required | one random genotype, for initialization |
| `descriptor_bounds(prob)` | required, unless categories | per-dimension `(lo, hi)` tuples |
| `descriptor_categories(prob)` | alternative to bounds | per-dimension collections of discrete descriptor values; descriptors are then tuples such as `(3, :ring)` |
| `evaluate_batch(prob, xs) -> Vector{Evaluation}` | optional | default: threaded loop over `evaluate` |

`QDProblem(f; init, bounds)` (or `categories` instead of `bounds`) wraps a plain function `f(x) -> (fitness,
descriptor)` so most users never subtype.

**`AbstractArchive{G,F,D}`**

| Function | Status | Meaning |
|---|---|---|
| `add!(archive, x, ev) -> AddResult` | required | try to insert one solution |
| `cell_index(archive, d)` | required | cell for descriptor `d`; must not allocate |
| `sample_elites(rng, archive, n) -> Vector{G}` | required | `n` parents, uniform over occupied cells by default |
| `elites(archive)` | required | iterator over `(genotype, evaluation)` of occupied cells |
| `Base.length(archive)` | required | number of occupied cells |
| `capacity(archive)` | required | number of cells |
| `add_batch!(archive, xs, evs)` | optional | default: loop over `add!`; the cell archives and `ThresholdArchive` compute every result against the archive as it was before the batch |
| `result_archive(archive)` | optional | the archive of the best solutions found, which `solve` returns and whose metrics a run records; default: the archive itself |
| `Base.empty!(archive)` | optional | |

Concrete archives: `GridArchive`, `CVTArchive`, `CategoricalArchive` (discrete
descriptors, numbers or labels), `ThresholdArchive` (CMA-MAE thresholds around
a cell archive, with a result archive of the best solutions), `UnstructuredArchive` (novelty-style,
distance threshold), later `ParetoGridArchive` (MOME).

Storage layout for cell-based archives (`GridArchive`, `CVTArchive`,
`CategoricalArchive`, subtypes of `CellArchive`, which holds the shared code) is struct-of-arrays:
`occupied::BitVector`, `fitness::Vector{F}`, `descriptors::Vector{D}`,
`genotypes::Vector{G}` (unassigned slots left `#undef` and guarded by
`occupied`), plus `occupied_idx::Vector{Int}` so a uniform parent draw is O(1).

**`AbstractVariation`**

| Function | Status | Meaning |
|---|---|---|
| `vary(rng, op, parent)` | required for mutations | one child |
| `vary(rng, op, p1, p2)` | required for crossovers | one child |
| `n_parents(op)` | required for crossovers | `1` (default) or `2`; `VariationEmitter` draws that many parents per child |

Defaults for `AbstractVector{<:Real}` genotypes: `GaussianMutation(σ)`,
`IsoLineVariation(σ_iso, σ_line)`, `PolynomialMutation(lower, upper)` for
bounded genotypes, and `UniformCrossover(proportion)`. A user with a custom genotype defines
`vary` for their own operator type.

**`AbstractEmitter`**

| Function | Status | Meaning |
|---|---|---|
| `ask(rng, emitter, archive) -> Vector{G}` | required | a batch of candidates |
| `tell!(emitter, xs, evs, results)` | optional | update internal state; default no-op |
| `batch_size(emitter)` | required | |

Concrete emitters: `VariationEmitter(op; batch_size)` (select parents
uniformly, apply `op`) and `CMAEmitter` (a search distribution whose update
is driven by a ranker, a selection rule and a restart rule, all types chosen by
dispatch; the optimizer is an `AbstractEvolutionStrategy`: `CMAES`, `SepCMAES`,
`LMMAES` or `OpenAIES`). An emitter with state may require `tell!` to follow the `ask` that
produced the batch.

**Algorithm and state** — CommonSolve.jl interface:

```julia
state = init(prob, alg; rng, budget, callback)   # -> QDState
step!(state)                                     # one ask/evaluate/add/tell round
solve!(state)                                    # run to budget
result = solve(prob, alg; kwargs...)             # init + solve!, returns QDResult
```

`QDState` holds the problem, a working copy of the algorithm's archive and
emitters, the RNG, the evaluation count, and the metrics history. Stepping
one iteration at a time is a supported use, not a debugging hack.

### The loop

```julia
function step!(state)
    for emitter in state.emitters          # tuple: unrolled, type-stable
        xs  = ask(state.rng, emitter, state.archive)
        evs = evaluate_batch(state.problem, xs)
        rs  = add_batch!(state.archive, xs, evs)
        tell!(emitter, xs, evs, rs)
        state.n_evals += length(xs)
    end
    record_metrics!(state)
    return state
end
```

Initialization fills the archive with `random_solution` draws before the
first step.

### Metrics

`qd_score(archive; offset)`, `coverage(archive)`, `max_fitness(archive)`,
`mean_fitness(archive)`. QD-score needs an offset so negative fitness values
do not reduce the score; the problem may supply a default via
`fitness_offset(prob)` (optional, default `0`).

---

## 4. Performance rules

These hold for code in `src/`. Tests in `test/` check them.

- No runtime dispatch in `step!`, `add!`, `cell_index`, `sample_elites`,
  `ask`, or `vary` for the built-in types. Check with `@inferred` and JET.
- `cell_index` and `add!` do not allocate for `isbits` genotypes. Check with
  `@allocated` after warm-up.
- Emitters are stored in a `Tuple`, not a `Vector{AbstractEmitter}`.
- Real-valued descriptors are `SVector`s from StaticArrays.jl. Discrete
  descriptors are tuples whose element types are those of the categories, so
  they are concretely typed too.
- Batch outputs are preallocated with types computed from the archive's type
  parameters, not grown with `push!` into an untyped container.
- No `@inbounds` without profiling evidence; see
  `.claude/rules/julia-inbounds.md`.
- Default `evaluate_batch` uses threads. Leave cores free for the user when
  running benchmarks.

---

## 5. Repository layout

```
QualityDiversity.jl/
├── Project.toml
├── CLAUDE.md                     # this file
├── src/
│   ├── QualityDiversity.jl       # module, exports, `public` declarations, includes
│   ├── interfaces.jl             # abstract types + documented function stubs
│   ├── evaluation.jl             # Evaluation, AddStatus, AddResult
│   ├── problems/
│   │   ├── qdproblem.jl          # QDProblem wrapper
│   │   └── benchmarks.jl         # Sphere, Rastrigin, later PlanarArm
│   ├── archives/
│   │   ├── cell_archive.jl       # CellArchive: shared storage, add!, metrics
│   │   ├── grid.jl
│   │   ├── nearest_centroid.jl   # k-d tree and k-means for CVTArchive
│   │   ├── cvt.jl
│   │   ├── categorical.jl
│   │   ├── threshold.jl
│   │   └── unstructured.jl
│   ├── variation/
│   │   ├── gaussian.jl
│   │   ├── isoline.jl
│   │   ├── polynomial.jl
│   │   └── crossover.jl         # UniformCrossover
│   ├── emitters/
│   │   ├── variation_emitter.jl
│   │   ├── evolution_strategy.jl # AbstractEvolutionStrategy and what the strategies share
│   │   ├── cma_es.jl             # the CMA-ES optimizer inside CMAEmitter
│   │   ├── sep_cma_es.jl
│   │   ├── lm_ma_es.jl
│   │   ├── openai_es.jl
│   │   ├── rankers.jl
│   │   └── cma_emitter.jl
│   ├── solve.jl                  # QDAlgorithm, QDState, QDResult, init/step!/solve!
│   ├── algorithms.jl             # MAPElites(), CMAME(), ... constructors
│   ├── metrics.jl
│   └── testing.jl                # QualityDiversity.Testing: interface conformance checks
├── ext/
│   ├── QualityDiversityMakieExt.jl        # archive heatmaps, metric curves
│   ├── QualityDiversityForwardDiffExt.jl  # differentiable QD
│   └── QualityDiversityPythonCallExt.jl   # batched evaluation through Python
├── test/
│   ├── runtests.jl
│   ├── test_archives.jl
│   ├── test_variation.jl
│   ├── test_emitters.jl
│   ├── test_solve.jl
│   ├── test_conformance.jl       # runs Testing checks on every built-in type
│   ├── test_inference.jl         # @inferred and allocation checks
│   └── test_quality.jl           # Aqua, ExplicitImports, JET
├── benchmark/
│   ├── Project.toml              # BenchmarkTools lives here, not in the package
│   ├── benchmarks.jl             # BenchmarkTools SUITE
│   ├── run.jl                    # runs the suite and prints a table
│   └── compare/                  # same problem in pyribs and QDax, for the comparison table
├── docs/                         # Documenter.jl
└── examples/
    ├── Project.toml              # Graphs.jl and JLArrays for the examples
    ├── rastrigin.jl              # continuous genotype
    ├── rastrigin_tutorial.jl     # step-by-step walkthrough
    ├── gpu_evaluation.jl         # optional: evaluate_batch on a GPU array type
    └── graphs.jl                 # structured genotype (proves the pitch)
```

`QualityDiversity.Testing` is a public submodule. A user who defines a new
problem, archive or emitter calls e.g.
`QualityDiversity.Testing.test_problem(MyProblem())` to check that it meets
the contract, is inferrable, and returns the right types. It lives in `src/`
(not `test/`) so users can call it; it depends only on the `Test` stdlib.

---

## 6. Roadmap

Work through phases in order. Each phase ends with a green test suite on
Julia LTS and the current release.

- [x] **Phase 1 — Contracts.** `interfaces.jl`, `evaluation.jl`, module file
      with exports and `public`. Docstrings for every abstract type and
      contract function. No implementations.
- [x] **Phase 2 — First end-to-end run.** `GridArchive`, `GaussianMutation`,
      `VariationEmitter`, `QDProblem`, `Rastrigin`, `init`/`step!`/`solve`,
      `MAPElites()`. Example `examples/rastrigin.jl` fills an archive.
- [x] **Phase 3 — Guarantees.** Metrics, `Testing` submodule, conformance
      tests, inference and allocation tests, Aqua/ExplicitImports/JET.
- [x] **Phase 4 — Benchmarks.** BenchmarkTools suite; comparison with pyribs
      on Rastrigin (same archive shape, batch size, evaluation budget).
- [x] **Phase 5 — Structured genotype.** `examples/graphs.jl` with a
      Graphs.jl genotype, custom `vary`, descriptor from graph statistics.
- [ ] **Phase 6 — More algorithms.** One at a time, easiest first. Each comes
      with tests, a conformance check from `Testing`, and a benchmark entry.
      The list follows what pyribs and QDax offer, minus what is out of scope.
  - [x] `IsoLineVariation` and `VariationEmitter` with two parents
  - [x] Bounded variation: `PolynomialMutation` (Deb's bounded form, as in
        NSGA-II and pymoo) with the bounds given to the operator, and
        `UniformCrossover` for vectors of any entry type (QDax calls it
        `polynomial_crossover`)
  - [x] `CVTArchive` (pyribs, QDax): centroids from k-means, nearest-centroid
        lookup with a k-d tree
  - [x] `CategoricalArchive` (pyribs): cells from discrete descriptor values
  - [x] `UnstructuredArchive` (QDax) / `ProximityArchive` (pyribs):
        novelty-style, a solution enters if it is far from its neighbours
  - [x] Threshold archive for CMA-MAE (pyribs `learning_rate`): `ThresholdArchive`,
        a separate type from `GridArchive` that wraps any cell archive
  - [x] `CMAEmitter` with a CMA-ES core, the two-stage improvement, improvement
        and objective rankers, filter and mu selection, and restart rules;
        `CMAME()` and `CMAMAE()`
  - [x] Random-direction rankers (CMA-ME-RD) and the two-stage objective ranker;
        further evolution strategies (sep-CMA-ES, LM-MA-ES, OpenAI-ES)
  - [ ] `ParetoGridArchive` and `MOME()` (QDax), with hypervolume metrics
  - [ ] `SlidingBoundariesArchive` (pyribs), `BanditScheduler` for choosing
        among emitters (pyribs), `DensityArchive` and `DNSArchive` (pyribs)
  - [ ] Stochastic problems and MAP-Elites with low spread (QDax `MELS`), after
        the stochastic-evaluation decision below
- [ ] **Phase 7 — Extensions and docs.** Makie (heatmaps, ECDF, CQD score),
      ForwardDiff (CMA-MEGA, OMG-MEGA), PythonCall; Bayesian-optimization
      emitter; Documenter site with tutorials.

GPU support is optional and never a dependency of the package. Scoring a batch
on a device works through `evaluate_batch` (`examples/gpu_evaluation.jl`,
tested with JLArrays on the CPU). A device-resident archive with batched
variation and insertion would be a separate archive type in an extension; it
waits until CMA-ME exists and there is a need for batches of 10⁴ or more.

Out of scope, as in the introduction: reinforcement-learning and
neural-policy methods in QDax (PGA-ME, QD-PG, DCRL, ME-ES on policies, DIAYN,
DADS, SMERL, PBT, TD3/SAC), AURORA (descriptors from a trained autoencoder),
and distributed MAP-Elites.

---

## Decisions

Settled; change them only with the author's agreement.

- **Fitness thresholds.** `GridArchive` is plain MAP-Elites: an empty cell
  accepts any solution, and a `NewCell` reports `improvement == fitness`.
  CMA-MAE thresholds (per-cell threshold with a learning rate) live in
  `ThresholdArchive`, which wraps a cell archive.
- **Batch results.** `add_batch!` of the cell archives and of
  `ThresholdArchive` computes the status and `improvement` of every solution
  against the archive as it was before the batch, as pyribs and QDax do, so that
  emitters rank candidates independently of the order in the batch. Of the
  solutions that enter a cell the best is kept (the first on a tie), so a cell
  archive ends up as if the solutions had been added one at a time.
  `ThresholdArchive` moves a threshold by the batch rule of the CMA-MAE paper.
  Archives without their own `add_batch!` (`UnstructuredArchive`) loop over `add!`.
- **Result archive.** A `ThresholdArchive` replaces the elite of a cell whenever
  a solution passes the threshold, so its elites can get worse (as in CMA-MAE
  and pyribs). It therefore also offers every solution to a plain archive, and
  `result_archive(archive)` returns it; `solve` returns it and a run records its
  metrics, while emitters read the `ThresholdArchive`. With `learning_rate=1` and
  `threshold_min=-Inf` the archive is its own result. `threshold_min=-Inf`
  requires `learning_rate=1`.
- **CMA emitters.** `CMAEmitter` works on real vectors. `tell!` has neither the
  random number generator nor the archive, so it only marks a restart; the next
  `ask` starts the new search from a random elite, drawn with the run's generator.
  Without `x0` the first search starts from a random elite as well, so the archive
  is filled before the first `ask`. `sigma0` has no default. `CMAME` defaults to the
  rules of CMA-ME (two-stage improvement ranking, filter selection, restart after a
  rejected batch) and `CMAMAE` to those of CMA-MAE (improvement ranking, mu
  selection, restart on convergence only) over a `ThresholdArchive`. The optimizer is
  chosen by the `strategy` keyword (a type, `CMAES` by default). A ranker may hold
  state, such as a random direction: an emitter keeps its own copy of its ranker, and
  calls `reset_ranker!(rng, ranker, archive)` whenever a search starts. A batch is flat
  when the whole ranking key of its best and its worst candidate agree, status
  included, as in pyribs and QDax. `PyribsCMAES` is CMA-ES with the rank-one weight `c1²`
  that pyribs computes, for reproducing results produced with pyribs; `CMAES` follows
  the textbook `c1`. `AcceptedRandomDirectionRanker` is the random direction ranking of
  the CMA-ME paper; the two-stage one is that of pyribs.
- **Outside-bounds descriptors.** `GridArchive` throws by default. The
  `outside` keyword takes `ThrowOutside()`, `ClampOutside()` or
  `RejectOutside()`, chosen by dispatch.
- **Grid limits.** `GridArchive` and `MAPElites` take an optional `bounds`
  that overrides `descriptor_bounds(prob)`.
- **Algorithm and problem.** `MAPElites(prob; dims)` takes the problem for its
  descriptor bounds; for a problem with `descriptor_categories`,
  `MAPElites(prob)` builds a `CategoricalArchive` and `dims` is an error. `init` checks that archive and problem agree on the
  number of descriptor dimensions.
- **Discrete descriptors.** `CategoricalArchive(categories)` takes one
  collection of allowed values per dimension, numbers or labels such as
  `Symbol`s. Its descriptor type is the tuple of their element types, e.g.
  `Tuple{Int,Symbol}`, so labels need no encoding. A problem declares
  `descriptor_categories` (or `QDProblem(...; categories)`) instead of
  `descriptor_bounds`. Archives that need geometry keep `SVector` descriptors.
- **Benchmark clipping.** `Rastrigin` clips the solution to `[-5.12, 5.12]`
  inside `evaluate`. Solution-space bounds for mutations are not part of the
  problem contract.

## Open decisions

Ask the author before resolving any of these.

- **Stochastic evaluation.** Direction: noisy problems receive an explicit RNG
  (`evaluate(rng, prob, x)`), so runs stay reproducible from the seed. To keep
  results independent of the thread count, the loop draws one seed per solution
  from the run's RNG before the threaded section and gives each solution its
  own `Xoshiro`. Deterministic problems keep `evaluate(prob, x)`; noisy ones
  opt in through a subtype such as `AbstractStochasticQDProblem`. Details are
  settled when it is implemented.
- **Phenotype storage.** In chemistry the mutated object (e.g. a reaction
  sequence) often differs from the described object (the product). The mapping
  from genotype to phenotype belongs to the problem; the open question is
  whether archives store the phenotype. Proposal: `Evaluation{F,D,P}` with an
  optional `phenotype::P` (default `Nothing`), stored alongside the genotype,
  with the two-argument `Evaluation(fitness, descriptor)` kept. Decide before
  the first example that needs it.

---

## Conventions

### Julia style

- Julia compat lower bound: **1.10 (LTS)**. Every dependency gets a
  `[compat]` entry whose lower bound works on 1.10. Run `Pkg.resolve()` after
  editing `Project.toml`.
- Annotate method arguments only as tightly as the implementation needs.
  `AbstractVector{<:Real}`, not `Vector{Float64}`. Annotate to control
  dispatch, not to document intent. An `AbstractArray` annotation promises
  that the code works for offset axes, views and GPU arrays; see
  `.claude/rules/julia-generic-indexing.md`.
- Use American spelling. Avoid jargon that experts in the field do not widely
  accept, and do not let technical prose read like a pitch deck.
- Parametric constructors: inner constructor with unconstrained value
  arguments; outer constructors compute type parameters and delegate inward,
  so `T(args...)`, `T{A}(args...)` and `T{A,B}(args...)` coerce identically.
- Use the short keyword form `f(; batch_size)` when the variable name matches.
- Prefer immutable `struct`s. A `mutable struct` needs a reason (for example
  `QDState`); mark fields that never change as `const`.
- Test error messages, not just exception types:
  `@test_throws "descriptor has 3 dimensions but the archive has 2" add!(...)`.
- Write docstrings for every exported or `public` name, with a runnable
  example where it helps. Test examples before committing them.

### Comments and commit messages

Comments, docstrings and commit messages describe what is true of the code
now, for a reader who has only the repository. No references to plans,
phases, roadmap items, chat sessions, or history ("previously", "for now").
Full rule: `.claude/rules/code-comments.md`. A hook (`.claude/hooks/`) flags
violations after each edit.

### Working with Julia

- Use Revise in a persistent session to avoid paying compilation on every
  run. If a Julia MCP server is configured, use it; otherwise use
  `julia --project` from the shell and keep runs few and purposeful.
- Activate the test environment with `TestEnv.activate()` (TestEnv, Revise,
  BenchmarkTools, JET live in the global environment, not in this project).
- `julia +lts` runs the LTS, `julia +release` the current release (juliaup).
  Run the full suite on both before declaring a phase done.
- When adding a dependency, choose it on its merits, not because it happens
  to be installed.
- Run `Pkg.test()` as the final check before a pull request; iterate with
  Revise and targeted includes.
- Do not add developer tools (BenchmarkTools, JET, Cthulhu, Revise) to the
  package's `[deps]`. Test-only dependencies go in `test/Project.toml`.

### Git and GitHub

- Subject lines ≤ 50 characters where possible, ≤ 72 at most.
- Do not post comments, issues or PRs on GitHub without the author's approval
  of the exact text, and do not represent the author on GitHub without
  consent.
- Reference an issue or PR that motivated a change in a code comment, not in
  the subject line. Close issues with `Fixes #xyz` in the body; for several,
  write `Fixes #abc; fixes #def` (ranges and comma lists do not work).
- Attribute agent assistance with an `Assisted-by:` trailer naming the model,
  not `Co-authored-by`.
- Do not commit planning documents, agent conversations or benchmark
  narratives. Add them to `.git/info/exclude`.
- Keep this file out of the registered package if the author decides so (add
  it to `.git/info/exclude`); until then it is committed.

### Skills available in this repository

The skills live in the user-level `~/.claude/skills` (see
[timholy/claude_config](https://github.com/timholy/claude_config)); none is
stored in this repository.

| Skill | Use |
|---|---|
| `/qd-new-component` | add a problem, archive, variation, emitter or algorithm that meets its contract, with tests |
| `/profile-performance` | find hot spots, runtime dispatch and allocations; propose fixes |
| `/qd-benchmark` | maintain the BenchmarkTools suite and the pyribs comparison |
| `/review-design`, `/review-api`, `/review-integration`, `/review-implement` | review the type hierarchy, public API and integration against this file, then implement the plan |
| `/freshen-aqua`, `/freshen-explicit-imports`, `/freshen-runic`, `/freshen-coverage`, `/freshen-docstrings`, `/freshen-docs`, `/freshen-generic-axes` | bring one quality check up to date |
| `/freshen-package` | run the `/freshen-*` skills in a sensible order |
| `/limit-struct-mutability` | make `mutable struct`s immutable or mark fields `const` |
