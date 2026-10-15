"""
    DeviceVariationEmitter(op; batch_size)

The emitter of MAP-Elites over a [`DeviceCellArchive`](@ref): it draws parents uniformly
from the occupied cells and applies the variation operator `op`, giving `batch_size`
children, all on the device. The children come back as a [`DeviceBatch`](@ref), which
`evaluate_batch` and `add_batch!` take without copying it to the host. A later `ask`
returns a new batch, so a batch stays valid after the next one is drawn.

`op` is a [`GaussianMutation`](@ref) or an [`IsoLineVariation`](@ref); its standard
deviations are converted to the element type of the archive. Each `ask` draws one
`UInt64` from the run's generator, and the kernels derive their random numbers from it
with the counter-based generator Philox4x32-10, keyed on the child and the entry, so a
run is reproducible from its seed on a given backend however the threads are scheduled.
These random numbers differ from those of a [`VariationEmitter`](@ref) on the host.
"""
struct DeviceVariationEmitter{V<:AbstractVariation} <: AbstractEmitter
    op::V
    batch_size::Int
    function DeviceVariationEmitter(op::V, batch_size::Integer) where {V<:AbstractVariation}
        check_device_variation(op)
        batch_size >= 1 ||
            throw(ArgumentError("batch_size must be at least 1, got $batch_size"))
        return new{V}(op, batch_size)
    end
end

DeviceVariationEmitter(op::AbstractVariation; batch_size::Integer) = DeviceVariationEmitter(op, batch_size)

check_device_variation(::GaussianMutation) = nothing
check_device_variation(::IsoLineVariation) = nothing

function check_device_variation(op::AbstractVariation)
    throw(ArgumentError("DeviceVariationEmitter supports GaussianMutation and IsoLineVariation, not $(nameof(typeof(op)))"))
end

batch_size(emitter::DeviceVariationEmitter) = emitter.batch_size

function ask(rng::AbstractRNG, emitter::DeviceVariationEmitter, archive::DeviceCellArchive{T}) where {T}
    length(archive) == 0 && throw(ArgumentError("cannot sample elites from an empty archive"))
    seed = rand(rng, UInt64)
    X = similar(archive.genotypes, T, solution_length(archive), emitter.batch_size)
    device_vary!(X, emitter.op, archive, seed)
    return DeviceBatch(X)
end

function ask(rng::AbstractRNG, emitter::DeviceVariationEmitter, archive::AbstractArchive)
    throw(ArgumentError("a DeviceVariationEmitter draws its parents from a device archive such as a DeviceGridArchive, not a $(nameof(typeof(archive)))"))
end

# Fills the columns of `X` with children of parents drawn from `archive`; defined in the
# KernelAbstractions extension.
device_vary!(X, op, archive, seed) = throw(ArgumentError(DEVICE_KERNELS_MESSAGE))
