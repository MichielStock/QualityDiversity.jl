# # Evaluating on a GPU
#
# QualityDiversity.jl runs on the CPU. When the objective is expensive and
# can be written as array operations, the batch of candidates can still be
# scored on a GPU: the loop calls `evaluate_batch(prob, xs)` once per round,
# and a problem type may define that method. The archive, the variation and
# the loop stay on the CPU, and only the batch of genotypes (one round of
# `batch_size` vectors) and the fitness and descriptor values (a few numbers
# per solution) cross to the device and back.
#
# This example is the Rastrigin function scored with array operations on an
# array type of your choice. Nothing here is specific to one GPU package: the
# operations are broadcasts and reductions, which every package supports.
#
# Run from the repository root, after the one-time setup
#
#     julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# With no GPU package installed, JLArrays stands in for a GPU. It is an array
# type that runs on the CPU, slowly, and shows that the example works:
#
#     julia --project=examples examples/gpu_evaluation.jl
#
# With a GPU, add its package to the examples environment and choose it:
#
#     julia --project=examples -e 'using Pkg; Pkg.add("CUDA")'
#     GPU_BACKEND=CUDA julia --project=examples examples/gpu_evaluation.jl
#
# `GPU_BACKEND` is `JLArray` (default), `CUDA`, `Metal`, `AMDGPU` or `oneAPI`.
# Only `JLArray` is tested; the others use the same array operations, but
# have not been run.
#
# A GPU pays off when the batch is large and the objective costly. For the
# cheap Rastrigin function the CPU is faster at small batches, because every
# round has to cross to the device and back.
#
# To keep the archive and the variation on the device as well, so that nothing
# crosses per round, pass `device=CuArray` (or another array type) to
# `MAPElites`; see `DeviceGridArchive`. A problem then scores the columns of a
# device matrix, and `GPURastrigin` below works with it unchanged.

module GPUExample

using JLArrays: JLArray
using QualityDiversity
using Random: Xoshiro
using StaticArrays: SVector

const LIMIT = 5.12

"""
    GPURastrigin(A, dim)

The Rastrigin benchmark over `dim` parameters, as in `Rastrigin`, with
`evaluate_batch` running on arrays of type `A` (`CuArray`, `MtlArray`,
`JLArray`, ...) in `Float32`. Genotypes are `Vector{Float32}` and fitness values
are `Float32`; descriptors are `Float64` as everywhere in the package.
"""
struct GPURastrigin{A} <: AbstractQDProblem
    dim::Int
    function GPURastrigin{A}(dim::Integer) where {A}
        dim >= 2 || throw(ArgumentError("GPURastrigin needs at least 2 dimensions, got $dim"))
        return new{A}(dim)
    end
end

GPURastrigin(A::Type, dim::Integer) = GPURastrigin{A}(dim)

QualityDiversity.random_solution(rng, prob::GPURastrigin) =
    (2f0 .* rand(rng, Float32, prob.dim) .- 1f0) .* Float32(LIMIT)

QualityDiversity.descriptor_bounds(::GPURastrigin) = ((-LIMIT, LIMIT), (-LIMIT, LIMIT))

QualityDiversity.fitness_offset(prob::GPURastrigin) = 41.0 * prob.dim

# The descriptor of the device code is computed in Float32 and may lie a rounding
# error outside the box in Float64.
descriptor(a, b) = SVector(clamp(Float64(a), -LIMIT, LIMIT), clamp(Float64(b), -LIMIT, LIMIT))

# One solution on the CPU: the reference that `evaluate_batch` is checked against.
function QualityDiversity.evaluate(prob::GPURastrigin, x::AbstractVector{<:Real})
    length(x) == prob.dim ||
        throw(ArgumentError("solution has $(length(x)) entries but GPURastrigin has dimension $(prob.dim)"))
    limit = Float32(LIMIT)
    half = prob.dim ÷ 2
    total = 0f0
    first_half = 0f0
    second_half = 0f0
    for i in 1:prob.dim
        xi = clamp(Float32(x[i]), -limit, limit)
        total += xi^2 - 10f0 * cos(2f0 * Float32(π) * xi)
        i <= half ? (first_half += xi) : (second_half += xi)
    end
    return Evaluation(-(10f0 * prob.dim + total),
                      descriptor(first_half / half, second_half / (prob.dim - half)))
end

# A batch on the device. The genotypes form a `dim × n` matrix, one solution
# per column, and everything below is a broadcast or a reduction over it.
function QualityDiversity.evaluate_batch(prob::GPURastrigin{A}, xs::AbstractVector) where {A}
    isempty(xs) && return Evaluation{Float32,SVector{2,Float64},Nothing}[]
    dim = prob.dim
    half = dim ÷ 2
    limit = Float32(LIMIT)
    all(==(dim), length.(xs)) ||
        throw(ArgumentError("every solution must have $dim entries"))
    X = A(Float32.(reduce(hcat, xs)))                       # dim × n, on the device
    X = clamp.(X, -limit, limit)
    terms = X .^ 2 .- 10f0 .* cos.(2f0 * Float32(π) .* X)
    fitness = -(10f0 * dim .+ sum(terms; dims=1))
    first_half = sum(view(X, 1:half, :); dims=1) ./ half
    second_half = sum(view(X, half+1:dim, :); dims=1) ./ (dim - half)
    f, a, b = vec(Array(fitness)), vec(Array(first_half)), vec(Array(second_half))
    return [Evaluation(f[i], descriptor(a[i], b[i])) for i in eachindex(f)]
end

# ## Choosing the array type

"""
    backend(name) -> Type

The array type of a GPU package, loaded on demand: `"JLArray"` (a CPU stand-in),
`"CUDA"`, `"Metal"`, `"AMDGPU"` or `"oneAPI"`.
"""
function backend(name::AbstractString)
    name == "JLArray" && return JLArray
    packages = Dict("CUDA" => (:CUDA, :CuArray), "Metal" => (:Metal, :MtlArray),
                    "AMDGPU" => (:AMDGPU, :ROCArray), "oneAPI" => (:oneAPI, :oneArray))
    haskey(packages, name) ||
        throw(ArgumentError("unknown backend $name; use JLArray, CUDA, Metal, AMDGPU or oneAPI"))
    package, array = packages[name]
    mod = try
        Base.require(Main, package)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("backend $name is not installed; add it with " *
                            "`julia --project=examples -e 'using Pkg; Pkg.add(\"$name\")'`"))
    end
    return getfield(mod, array)
end

# ## Running it

function main(A, name; budget=360_000, batch_size=10_000)
    prob = GPURastrigin(A, 20)
    alg = MAPElites(prob; dims=(100, 100), batch_size, genotype=Vector{Float32}, fitness=Float32,
                    variation=GaussianMutation(0.5f0))
    solve(prob, alg; budget=batch_size, rng=Xoshiro(0))      # compile
    seconds = @elapsed result = solve(prob, alg; budget, rng=Xoshiro(1))
    println("array type:   ", A, name == "JLArray" ? "  (a CPU stand-in, not a GPU)" : "")
    println("batch size:   ", batch_size)
    println("evaluations:  ", result.n_evals, " in ", round(seconds; digits=2), " s (",
            round(Int, result.n_evals / seconds), " per second)")
    println("coverage:     ", round(coverage(result.archive); digits=3))
    println("max fitness:  ", round(max_fitness(result.archive); digits=1))
    return result
end

end # module

if abspath(PROGRAM_FILE) == (@__FILE__)
    name = get(ENV, "GPU_BACKEND", "JLArray")
    A = GPUExample.backend(name)
    # the GPU package was loaded just now, so run in the newest world
    Base.invokelatest(GPUExample.main, A, name)
end
