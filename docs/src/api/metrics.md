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
QualityDiversity.CQDScore
```

## Several objectives

```@docs
hypervolume
QualityDiversity.max_hypervolume
global_hypervolume
QualityDiversity.max_scores
```

## Plotting

Loading Makie (for example `using CairoMakie`) adds plot recipes for archives; see
[Plotting archives](../tutorials/heatmap.md).

```@docs
Main.QualityDiversityMakieExt
```
