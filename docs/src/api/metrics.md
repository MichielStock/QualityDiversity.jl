```@meta
CurrentModule = QualityDiversity
```

# Metrics

```@docs
qd_score
coverage
max_fitness
mean_fitness
cqd_score
CQDScore
```

## Several objectives

```@docs
hypervolume
max_hypervolume
global_hypervolume
max_scores
```

## Plotting

Loading Makie (for example `using CairoMakie`) adds plot recipes for archives; see
[Plotting archives](../tutorials/heatmap.md).

```@docs
Main.QualityDiversityMakieExt
```
