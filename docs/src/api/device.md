```@meta
CurrentModule = QualityDiversity
```

# Device archives and emitters

These types run MAP-Elites and CMA-ME with the archive in the arrays of a GPU; see the
[GPU guide](../guide/gpu.md). Their kernels need KernelAbstractions.jl, which the GPU
packages load.

## Archives

```@docs
QualityDiversity.DeviceCellArchive
DeviceGridArchive
DeviceCVTArchive
```

## Batches

```@docs
QualityDiversity.DeviceBatch
QualityDiversity.EvaluationBatch
QualityDiversity.AddResultBatch
```

## Emitters

```@docs
QualityDiversity.DeviceVariationEmitter
QualityDiversity.DeviceCMAEmitter
```
