# Benchmarks

`benchmarks.jl` defines a BenchmarkTools `SUITE` for the hot path: problem
evaluation, variation, the grid archive (`cell_index`, `add!` for a new,
improved and rejected solution, sampling, metrics), the emitter, one
`step!`, and a full `solve`.

```
julia --threads=1 benchmark/run.jl
```

The first run installs the benchmark environment and links the package. Leave
cores free while it runs: the default `evaluate_batch` uses every thread Julia
was started with.

`pyribs_compare/` compares a full MAP-Elites run with pyribs; see its README
for the setup and results.
