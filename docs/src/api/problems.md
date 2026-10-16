```@meta
CurrentModule = QualityDiversity
```

# Problems

A problem scores a solution and describes it. Most users wrap a function with
[`QDProblem`](@ref) or [`StochasticQDProblem`](@ref); subtyping
[`AbstractQDProblem`](@ref) is for problems that need their own type, for example to
define [`evaluate_batch`](@ref QualityDiversity.evaluate_batch).

## Interface

```@docs
AbstractQDProblem
AbstractStochasticQDProblem
Evaluation
QualityDiversity.evaluate
QualityDiversity.evaluate_batch
QualityDiversity.random_solution
QualityDiversity.descriptor_bounds
QualityDiversity.descriptor_categories
QualityDiversity.fitness_offset
QualityDiversity.reference_point
QualityDiversity.evaluations_per_solution
```

## Wrappers

```@docs
QDProblem
StochasticQDProblem
Sampled
SampledSpread
QualityDiversity.DescriptorSamples
QualityDiversity.Expectation
QualityDiversity.expectation
Average
Median
```

## Benchmarks

```@docs
Rastrigin
Sphere
BiSphere
```
