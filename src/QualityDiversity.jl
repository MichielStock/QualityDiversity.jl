"""
    QualityDiversity

Quality-diversity optimization for any genotype that can be written as a Julia
type.

Five roles make up an algorithm: a problem ([`AbstractQDProblem`](@ref)), an
archive ([`AbstractArchive`](@ref)), variation operators
([`AbstractVariation`](@ref)), emitters ([`AbstractEmitter`](@ref)), and the
algorithm that combines them ([`QDAlgorithm`](@ref)). Each role is an abstract
type with a documented set of required and optional functions.

Interface functions such as `evaluate` and `add!` are `public` but not
exported. Extend them with qualified names:

```julia
QualityDiversity.evaluate(p::MyProblem, x) = ...
```
"""
module QualityDiversity

using CommonSolve: CommonSolve, solve
using Compat: @compat

include("evaluation.jl")
include("interfaces.jl")
include("metrics.jl")

export Evaluation, AddStatus, AddResult
export AbstractQDProblem, AbstractArchive, AbstractVariation, AbstractEmitter
export QDAlgorithm
export solve
export qd_score, coverage, max_fitness, mean_fitness

@compat public Rejected, Improved, NewCell
@compat public evaluate, evaluate_batch, random_solution, descriptor_bounds, fitness_offset
@compat public add!, add_batch!, cell_index, sample_elites, elites, capacity
@compat public vary
@compat public ask, tell!, batch_size

end
