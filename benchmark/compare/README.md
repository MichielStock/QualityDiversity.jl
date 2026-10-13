# QualityDiversity.jl, pyribs and QDax on Rastrigin

MAP-Elites on the 20-dimensional Rastrigin function, run with the same
settings in three libraries.

| setting | value |
|---|---|
| problem | Rastrigin, 20 parameters, clipped to [-5.12, 5.12]; descriptor = mean of the first and of the last 10 clipped parameters |
| archive | 100 × 100 grid over [-5.12, 5.12]², plain MAP-Elites (no thresholds) |
| mutation | Gaussian, σ = 0.5, no clipping |
| initial population | one batch of uniform random solutions |
| budget | 360 000 evaluations, a multiple of every batch size |
| batch sizes | 36, 100, 1000, 10 000 |
| repeats | 21 timed runs per batch size, one seed each, after a warm-up run (3 for QDax with its own cell lookup, see below) |
| hardware | every library pinned to one core of a 4-vCPU cloud container (Intel Xeon @ 2.1 GHz) with `taskset -c 1` |
| software | Julia 1.13.1 (1 thread), Python 3.13, pyribs 0.12.0, numpy 2.5.4, QDax 0.5.0, jax 0.11.2 (CPU) |

`compare.jl`, `compare.py` and `compare_qdax.py` define the problem
separately. Each prints the same columns. `compare.jl` and `compare.py` print
the fitness and descriptor of one fixed solution first, so the two can be
checked against each other:

```
Julia   reference fitness -407.77649216868974 descriptor (-3.0450526315789466, 3.0450526315789475)
pyribs  reference fitness -407.7764921686897  descriptor (-3.0450526315789475, 3.045052631578947)
```

Time covers building the archive, the initial batch and the rest of the
budget; compilation is excluded. Julia is timed with BenchmarkTools, Python
with `time.perf_counter`. "Middle half" is the range from the first to the
third quartile of the run times.

## How each library is used

- **QualityDiversity.jl:** `MAPElites` with a `GaussianMutation` and the
  default threaded `evaluate_batch` (one thread here). The archive records
  its metrics after every step. It runs in `Float64` with the package's
  `Rastrigin`, and in `Float32` with the same function written as a
  `QDProblem` with `Float32` arithmetic, `genotype=Vector{Float32}`,
  `fitness=Float32` and `GaussianMutation(0.5f0)`. The archive then holds
  `Float32` genotypes and fitness values and the mutation produces `Float32`
  children, with no change to the package.
- **pyribs:** `GridArchive`, `GaussianEmitter` and `Scheduler`. The objective
  is vectorized numpy, the fastest way to write it in Python.
- **QDax:** the whole run, including the initial batch, is one `jax.jit`
  function that scans `MAPElites.scan_update`, as the QDax documentation
  recommends. The objective and the mutation are jitted array code, and QDax's
  own QD score, maximum fitness and coverage are computed after every step.
  QDax is written for accelerators and these runs use the CPU. It works in
  `float32` by default; `JAX_ENABLE_X64=1` gives `float64`, which is what the
  other two libraries use.

QDax finds the cell of a descriptor by comparing it with every centroid
(`get_cells_indices`). That works for any tessellation and is cheap on a GPU,
but on a CPU it costs batch size × number of cells and takes almost all of the
run time: at batch size 1000 it is 148 ms per step, against under 1 ms for
the objective, the mutation, the selection and the metrics together. So QDax
is reported twice:

- **QDax, centroid search** is QDax exactly as documented. It takes about 45 s
  whatever the batch size, so each batch size has only 3 timed runs; the run
  times differ by under 5%.
- **QDax, grid index** replaces that one function with the arithmetic index of a
  regular grid (`QDAX_GRID_LOOKUP=1`). A check at start-up confirms it assigns
  the same cell as QDax's own lookup to 20 000 random descriptors (0 differ).
  Nothing else changes. This is the fair CPU comparison: it is what someone
  who needs QDax to be fast on a CPU would write.

## Results

Median time in seconds for the whole budget, one core:

| batch size | QualityDiversity.jl, float64 | QualityDiversity.jl, float32 | pyribs | QDax, grid index, float32 | QDax, grid index, float64 | QDax, centroid search, float32 |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.348 | 0.335 | 3.659 | 0.853 | 1.371 | 44.7 |
| 100 | 0.317 | 0.280 | 1.658 | 0.451 | 0.805 | 43.3 |
| 1000 | 0.313 | 0.260 | 0.590 | 0.168 | 0.332 | 42.8 |
| 10 000 | 0.317 | 0.278 | 0.568 | 0.160 | 0.321 | 46.8 |

Time of QDax divided by time of QualityDiversity.jl at the same precision, so
values above 1 mean QualityDiversity.jl is faster:

| batch size | float32 (QDax, grid index) | float64 (QDax, grid index) | QDax, centroid search, against float32 |
|---:|---:|---:|---:|
| 36 | 2.5 | 3.9 | 133 |
| 100 | 1.6 | 2.5 | 155 |
| 1000 | 0.65 | 1.06 | 164 |
| 10 000 | 0.58 | 1.01 | 168 |

Time of pyribs divided by time of QualityDiversity.jl in `float64`:
10.5, 5.2, 1.9 and 1.8 for batch sizes 36, 100, 1000 and 10 000.

In short: QualityDiversity.jl is faster than QDax on a CPU up to a batch size
of about 100, and QDax with the grid index is faster from 1000 up. In
`float32`, the precision QDax uses by default, QDax is about 1.5 to 1.7 times
faster at batch sizes 1000 and 10 000. In `float64` the two are level from
batch size 1000. `Float32` makes QualityDiversity.jl 4% to 17% faster, less
than the factor of two QDax gains, because Julia's scalar `cos` is not much
faster in `Float32` while XLA vectorizes it. pyribs is slower than both at
every batch size.

QualityDiversity.jl takes about the same time at every batch size (0.26 to
0.35 s, over a million evaluations per second), because each solution costs
the same work: one mutation, one evaluation and one insertion. The libraries
that vectorize a whole batch gain from larger batches.

### Full tables

**QualityDiversity.jl, float64, 1 thread**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.314 | 0.348 | 0.333–0.375 | 1 034 400 | 0.986 | -130.5 | 4 779 398 |
| 100 | 0.268 | 0.317 | 0.305–0.338 | 1 134 354 | 0.986 | -133.7 | 4 786 431 |
| 1000 | 0.282 | 0.313 | 0.291–0.348 | 1 149 679 | 0.986 | -133.2 | 4 796 417 |
| 10 000 | 0.262 | 0.317 | 0.288–0.345 | 1 134 320 | 0.942 | -130.6 | 4 614 685 |

**QualityDiversity.jl, float32, 1 thread**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.287 | 0.335 | 0.311–0.353 | 1 073 332 | 0.989 | -132.2 | 4 777 084 |
| 100 | 0.252 | 0.280 | 0.270–0.288 | 1 287 608 | 0.986 | -133.5 | 4 779 398 |
| 1000 | 0.238 | 0.260 | 0.251–0.273 | 1 382 371 | 0.986 | -133.8 | 4 793 316 |
| 10 000 | 0.242 | 0.278 | 0.265–0.292 | 1 293 366 | 0.945 | -131.1 | 4 617 796 |

**pyribs** (`OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`)

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 3.370 | 3.659 | 3.565–3.828 | 98 386 | 0.988 | -136.5 | 4 790 761 |
| 100 | 1.443 | 1.658 | 1.607–1.743 | 217 188 | 0.988 | -134.0 | 4 796 704 |
| 1000 | 0.544 | 0.590 | 0.574–0.599 | 609 720 | 0.986 | -132.5 | 4 803 731 |
| 10 000 | 0.522 | 0.568 | 0.550–0.584 | 634 046 | 0.945 | -135.2 | 4 630 502 |

**QDax, grid index, float32**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.732 | 0.853 | 0.815–0.881 | 422 009 | 0.987 | -131.1 | 4 780 165 |
| 100 | 0.404 | 0.451 | 0.425–0.474 | 798 945 | 0.988 | -136.9 | 4 789 608 |
| 1000 | 0.146 | 0.168 | 0.161–0.184 | 2 140 452 | 0.985 | -135.3 | 4 788 907 |
| 10 000 | 0.143 | 0.160 | 0.148–0.166 | 2 245 315 | 0.944 | -127.8 | 4 623 051 |

**QDax, grid index, float64**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.272 | 1.371 | 1.343–1.414 | 262 609 | 0.986 | -133.5 | 4 780 745 |
| 100 | 0.723 | 0.805 | 0.769–0.818 | 447 417 | 0.985 | -133.3 | 4 771 358 |
| 1000 | 0.300 | 0.332 | 0.318–0.343 | 1 083 718 | 0.986 | -131.9 | 4 795 016 |
| 10 000 | 0.287 | 0.321 | 0.310–0.352 | 1 119 774 | 0.942 | -131.1 | 4 613 920 |

**QDax, centroid search, float32** (3 runs per row)

| batch size | min s | median s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 44.259 | 44.722 | 8 050 | 0.985 | -139.7 | 4 748 242 |
| 100 | 43.191 | 43.289 | 8 316 | 0.986 | -141.1 | 4 789 824 |
| 1000 | 41.725 | 42.759 | 8 419 | 0.987 | -134.7 | 4 804 787 |
| 10 000 | 45.567 | 46.773 | 7 697 | 0.948 | -124.1 | 4 629 452 |

Coverage and QD score agree across the libraries at each batch size to within
the spread between seeds. At batch size 10 000 every library fills fewer cells
(about 0.94), because the budget allows only 36 rounds.

## What the numbers do not say

- **No GPU.** QDax is built for accelerators, where the work per step is
  parallel across thousands of solutions. These runs use one CPU core and say
  nothing about QDax on a GPU or TPU, where it would be much faster than any of
  these numbers.
- **Cheap objective.** Rastrigin takes about 200 ns per solution. A slower
  objective shrinks the ratios for every library, and an objective that cannot
  be written as jittable array code cannot be used in QDax at all.
- **The QDax grid index is my change.** It is a small, checked replacement of
  one function and not part of QDax. Both QDax variants are reported so the
  difference is visible.
- **`float32` against `float64`.** Compare like with like: QualityDiversity.jl
  in `float32` with QDax in `float32`, and `float64` with `float64`. pyribs
  runs in `float64`.
- **Spread.** The run-to-run spread on this shared machine is about ±10%
  within a batch size, so close values, such as QualityDiversity.jl against
  QDax in `float64` at 1000 and 10 000, are level.
- **Threads.** Everything is pinned to one core. Neither the threaded
  `evaluate_batch` of QualityDiversity.jl nor multi-core JAX is measured.

## Reproducing

```
# Julia
taskset -c 1 julia --threads=1 benchmark/compare/compare.jl

# pyribs and QDax
python -m venv venv && venv/bin/pip install -r benchmark/compare/requirements.txt
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
taskset -c 1 venv/bin/python benchmark/compare/compare.py
QDAX_GRID_LOOKUP=1 taskset -c 1 venv/bin/python benchmark/compare/compare_qdax.py
JAX_ENABLE_X64=1 QDAX_GRID_LOOKUP=1 taskset -c 1 venv/bin/python benchmark/compare/compare_qdax.py
REPEATS=3 taskset -c 1 venv/bin/python benchmark/compare/compare_qdax.py   # about 15 minutes
```
