```@meta
CurrentModule = QualityDiversity
```

# Archives

An archive stores the solutions found so far, and decides which of them to keep.

## Interface

```@docs
AbstractArchive
AddStatus
AddResult
QualityDiversity.add!
QualityDiversity.add_batch!
QualityDiversity.cell_index
QualityDiversity.sample_elites
QualityDiversity.elites
QualityDiversity.capacity
QualityDiversity.result_archive
```

## Cell archives

```@docs
QualityDiversity.CellArchive
GridArchive
SlidingBoundariesArchive
QualityDiversity.boundaries
CVTArchive
QualityDiversity.centroids
CategoricalArchive
```

## Descriptors outside the bounds

```@docs
OutsidePolicy
ThrowOutside
ClampOutside
RejectOutside
```

## Archives around a cell archive

```@docs
ThresholdArchive
QualityDiversity.thresholds
SpreadArchive
QualityDiversity.spreads
DensityArchive
QualityDiversity.density
GradientArchive
```

## Archives without cells

```@docs
UnstructuredArchive
DNSArchive
```

## Several objectives

```@docs
ParetoGridArchive
num_solutions
DuplicatePolicy
RejectDuplicates
KeepDuplicates
FullFrontPolicy
RejectNewcomer
ReplaceCrowded
```
