```@meta
CurrentModule = QualityDiversity
```

# Running on a GPU

Everything runs on the CPU unless you ask for a device, and the package does not
depend on any GPU package. There are two ways to use a GPU, depending on what is
expensive.

## Evaluating a batch on a GPU

When the evaluation is expensive and the rest of the run is cheap, keep the archive on
the CPU and evaluate each batch on the GPU. The loop calls
[`evaluate_batch`](@ref QualityDiversity.evaluate_batch) once per round, and a problem
type can define it: copy the batch to the device, compute with array operations or a
kernel, and copy the fitness and descriptors back.

```julia
function QualityDiversity.evaluate_batch(prob::MyProblem, xs::AbstractVector)
    X = CuArray(reduce(hcat, xs))                     # one column per solution
    fitness, descriptors = my_gpu_scoring(X)          # array code on the device
    f, d = Array(fitness), Array(descriptors)
    return [Evaluation(f[j], SVector(d[1, j], d[2, j])) for j in eachindex(f)]
end
```

`examples/gpu_evaluation.jl` does this for the Rastrigin function with any array type
(`CuArray`, `ROCArray`, `MtlArray`, `oneArray`), and is tested with JLArrays, an array
type that runs on the CPU.

## MAP-Elites and CMA-ME on a device

When evaluation is cheap, the time goes into the archive and the variation, and moving
a batch to the GPU and back every round costs more than it saves. Then the whole run
can stay on the device:

```julia
using CUDA, QualityDiversity
using Random: Xoshiro

prob = Rastrigin(100)
alg = MAPElites(prob; dims=(100, 100), batch_size=10_000, device=CuArray,
                genotype=Vector{Float32}, fitness=Float32)
result = solve(prob, alg; budget=10_000_000, rng=Xoshiro(1))
```

With `device`, [`MAPElites`](@ref) builds a [`DeviceGridArchive`](@ref), whose cells are
in arrays of that type, and a [`DeviceVariationEmitter`](@ref), which draws parents and
mutates them in a kernel. [`CMAME`](@ref) accepts `device` too and builds a
[`DeviceCMAEmitter`](@ref), many CMA-ES searches whose candidates are evaluated and
added together. For a CVT archive, build a [`DeviceCVTArchive`](@ref) and pass it to
`MAPElites(archive)`.

- **Kernels.** The kernels are written with KernelAbstractions.jl, which CUDA.jl,
  AMDGPU.jl and oneAPI.jl load. `device=Array` runs the same kernels on the CPU after
  `using KernelAbstractions`, which is how the tests run them.
- **Evaluation.** By default the problem's `evaluate` runs in a kernel, one solution
  per thread, so it must compile for the device: no allocation and no error messages
  built from values. [`Rastrigin`](@ref), [`Sphere`](@ref) and a [`QDProblem`](@ref)
  with a simple function work. Otherwise define `evaluate_batch(prob,
  xs::DeviceBatch)` and work on `parent(xs)`, a matrix with one solution per column.
- **Precision.** Consumer GPUs compute `Float64` many times more slowly than `Float32`,
  so pass `genotype=Vector{Float32}` and `fitness=Float32`. The cell of a descriptor is
  always computed in `Float64`, so that device and CPU archives assign the same cells;
  a GPU without `Float64`, such as an Apple GPU, cannot run the device archives.
- **Reproducibility.** A run is reproducible from its seed on a given device, whatever
  the scheduling of the threads, but its random numbers differ from those of the CPU
  path.
- **Results.** `GridArchive(archive)` and `CVTArchive(archive)` copy a device archive
  to the CPU, for plotting and analysis.

The device archives support MAP-Elites with [`GaussianMutation`](@ref) and
[`IsoLineVariation`](@ref), and CMA-ME. Stochastic problems, phenotypes and the other
archives run on the CPU.

## When it pays off

A GPU pays off for large batches. On the benchmark of the
[comparison](comparison.md), MAP-Elites with batches of 100 or fewer is faster on one
CPU core than on a GPU, and with batches of 10 000 the GPU is about 26 times faster.
A device CMA-ME is slower than the CPU with the five searches of the CMA-ME paper and
faster with hundreds.
