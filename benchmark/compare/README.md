# QualityDiversity.jl and pyribs on Rastrigin

MAP-Elites on the 20-dimensional Rastrigin function, run with the same
settings in both libraries.

| setting | value |
|---|---|
| problem | Rastrigin, 20 parameters, clipped to [-5.12, 5.12]; descriptor = mean of the first and of the last 10 clipped parameters |
| archive | 100 × 100 grid over [-5.12, 5.12]², plain MAP-Elites (no thresholds) |
| mutation | Gaussian, σ = 0.5, no clipping |
| initial population | one batch of uniform random solutions |
| budget | 360 000 evaluations, a multiple of every batch size |
| batch sizes | 36, 100, 1000 |
| repeats | 21 timed runs per batch size, one seed each, after a warm-up run; Julia timed with BenchmarkTools (`evals=1`), Python with `time.perf_counter` |

`compare.jl` and `compare.py` define the problem separately and print the
fitness and descriptor of one fixed solution first, so the two sides can be
checked against each other:

```
Julia   reference fitness -407.77649216868974 descriptor (-3.0450526315789466, 3.0450526315789475)
pyribs  reference fitness -407.7764921686897  descriptor (-3.0450526315789475, 3.045052631578947)
```

## Results

Run on a 4-vCPU cloud container (Intel Xeon @ 2.1 GHz), Julia 1.13.1,
Python 3.13, pyribs 0.12.0, numpy 2.5.4. Time covers building the archive and
running the whole budget; compilation is excluded. pyribs uses vectorized
numpy for the objective, the fastest way to write it in Python. "Middle half"
is the range from the first to the third quartile of the 21 run times.

**Julia, 1 thread**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.311 | 0.368 | 0.361–0.410 | 977 033 | 0.986 | -130.5 | 4 779 398 |
| 100 | 0.298 | 0.346 | 0.329–0.356 | 1 041 910 | 0.986 | -133.7 | 4 786 431 |
| 1000 | 0.275 | 0.294 | 0.289–0.308 | 1 223 565 | 0.986 | -133.2 | 4 796 417 |

**pyribs, 1 thread** (`OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`)

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 3.369 | 3.695 | 3.604–3.884 | 97 423 | 0.988 | -136.5 | 4 790 761 |
| 100 | 1.561 | 1.700 | 1.622–1.841 | 211 805 | 0.988 | -134.0 | 4 796 704 |
| 1000 | 0.524 | 0.593 | 0.555–0.615 | 607 397 | 0.986 | -132.5 | 4 803 731 |

Comparing medians, QualityDiversity.jl is about 10× faster at batch size 36,
4.9× at 100 and 2.0× at 1000; comparing minimum times gives 10.8×, 5.2× and
1.9×. Both reach the same coverage and QD score to within the spread between
seeds.

**Julia, 2 threads**

| batch size | min s | median s | middle half s |
|---:|---:|---:|---:|
| 36 | 0.260 | 0.308 | 0.290–0.325 |
| 100 | 0.202 | 0.247 | 0.235–0.266 |
| 1000 | 0.218 | 0.285 | 0.252–0.299 |

Threading helps at batch sizes 36 and 100 but not at 1000, and the 2-thread
times are noisier. This is a shared container with 4 vCPUs, so the threaded
rows do not show how the threaded `evaluate_batch` scales. Evaluating one
solution costs about 200 ns, which is too little work to split across threads
at these batch sizes.

## What the numbers do not say

- Rastrigin is cheap to evaluate, which is where library overhead shows most.
  A slow objective shrinks the ratios.
- The two libraries do not do identical bookkeeping. QualityDiversity.jl
  records `qd_score`, `coverage`, `max_fitness` and `mean_fitness` after every
  step; `GridArchive` keeps running totals, so this costs a few nanoseconds.
  pyribs computes its statistics while adding.
- The run-to-run spread on this machine is about ±10% within a batch size,
  comparable to the differences between the Julia batch sizes, so the order of
  Julia rows that are close together is not reliable. Re-run on your own
  hardware before quoting the ratios.

## Reproducing

```
# Julia
julia --threads=1 benchmark/compare/compare.jl

# Python
python -m venv venv && venv/bin/pip install -r benchmark/compare/requirements.txt
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 venv/bin/python benchmark/compare/compare.py
```
