```@meta
CurrentModule = QualityDiversity
```

# Variation

A variation operator makes a child from one or two parents. The operators below work on
vectors of real numbers; a genotype of another type gets a `vary` method for an operator
of its own, as in the [graph tutorial](../tutorials/graphs.md).

## Interface

```@docs
AbstractVariation
QualityDiversity.vary
QualityDiversity.n_parents
```

## Operators

```@docs
GaussianMutation
IsoLineVariation
PolynomialMutation
UniformCrossover
```
