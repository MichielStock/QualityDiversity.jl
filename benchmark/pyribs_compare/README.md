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
| repeats | 5 seeds after one warm-up run; the median is reported |

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
numpy for the objective, the fastest way to write it in Python.

**Julia, 1 thread**

| batch size | median s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|
| 36 | 0.341 | 1 054 594 | 0.985 | -129.4 | 4 790 731 |
| 100 | 0.381 | 944 679 | 0.988 | -128.0 | 4 794 731 |
| 1000 | 0.296 | 1 216 350 | 0.984 | -127.7 | 4 790 692 |

**pyribs, 1 thread** (`OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`)

| batch size | median s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|
| 36 | 3.853 | 93 430 | 0.988 | -137.5 | 4 781 722 |
| 100 | 1.864 | 193 120 | 0.990 | -133.0 | 4 800 441 |
| 1000 | 0.630 | 571 617 | 0.985 | -123.5 | 4 795 662 |

QualityDiversity.jl is about 11× faster at batch size 36, 4.9× at 100 and
2.1× at 1000. Both reach the same coverage and QD score to within the spread
between seeds. With 2 Julia threads the median times are 0.291 s, 0.249 s and
0.240 s; the threaded `evaluate_batch` gains little at small batches, because
evaluating one solution costs about 200 ns.

## What the numbers do not say

- Rastrigin is cheap to evaluate, which is where library overhead shows most.
  A slow objective shrinks the ratios.
- The two libraries do not do identical bookkeeping. QualityDiversity.jl
  records `qd_score`, `coverage`, `max_fitness` and `mean_fitness` after every
  step; `GridArchive` keeps running totals, so this costs a few nanoseconds.
  pyribs computes its statistics while adding.
- One machine, one run of five seeds. Run-to-run variation is visible in the
  gap between the minimum and the median time. Re-run on your own hardware
  before quoting the ratios.

## Reproducing

```
# Julia
julia --threads=1 benchmark/pyribs_compare/compare.jl

# Python
python -m venv venv && venv/bin/pip install -r benchmark/pyribs_compare/requirements.txt
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 venv/bin/python benchmark/pyribs_compare/compare.py
```
