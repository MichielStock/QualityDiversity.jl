"""
    QualityDiversityKernelAbstractionsExt

The kernels of the device archives ([`DeviceCellArchive`](@ref QualityDiversity.DeviceCellArchive))
and of [`DeviceVariationEmitter`](@ref QualityDiversity.DeviceVariationEmitter), written with
KernelAbstractions.jl so that they run on its backends with `Float64`, which the cell
lookup needs: CUDA, ROCm, oneAPI, and the CPU for `Array`.
"""
module QualityDiversityKernelAbstractionsExt

using Atomix: @atomic
using KernelAbstractions: KernelAbstractions, @kernel, @index, get_backend
using QualityDiversity: QualityDiversity, AbstractQDProblem, DeviceCellArchive, DeviceGridArchive, DeviceCVTArchive,
    AddResultBatch, QDProblem,
    GaussianMutation, IsoLineVariation, ThrowOutside, RejectOutside, Rejected, Improved, NewCell
using StaticArrays: SVector

const KA = KernelAbstractions
const QD = QualityDiversity

const REJECTED = UInt8(Integer(Rejected))
const IMPROVED = UInt8(Integer(Improved))
const NEW_CELL = UInt8(Integer(NewCell))
const NONE = typemax(Int32)

# Kernels on the CPU backend may run asynchronously; on a GPU the launches and copies of
# one task are ordered and nothing needs to wait.
sync!(backend::KA.CPU) = KA.synchronize(backend)
sync!(backend) = nothing

# ── Philox4x32-10 (Salmon et al., 2011) ─────────────────────────────────────────────
# A counter-based generator: the four words for a counter and a key are a fixed
# function of the two, so a thread computes its random numbers from its own indices.

@inline function philox(c0::UInt32, c1::UInt32, c2::UInt32, c3::UInt32, k0::UInt32, k1::UInt32)
    for round in 1:10
        if round > 1
            k0 += 0x9e3779b9
            k1 += 0xbb67ae85
        end
        p0 = UInt64(0xd2511f53) * UInt64(c0)
        p1 = UInt64(0xcd9e8d57) * UInt64(c2)
        c0, c1, c2, c3 = ((p1 >> 32) % UInt32) ⊻ c1 ⊻ k0, p1 % UInt32,
                         ((p0 >> 32) % UInt32) ⊻ c3 ⊻ k1, p0 % UInt32
    end
    return (c0, c1, c2, c3)
end

# The streams that the counters of the variation kernels draw from; a child `j` and an
# entry `i` complete the counter.
const STREAM_ENTRY = 0x00000000
const STREAM_PARENT = 0x00000001
const STREAM_SECOND_PARENT = 0x00000002
const STREAM_LINE = 0x00000003

@inline function random_words(key::NTuple{2,UInt32}, stream::UInt32, j, i)
    return philox(j % UInt32, i % UInt32, stream, 0x00000000, key[1], key[2])
end

key_of(seed::UInt64) = (seed % UInt32, (seed >> 32) % UInt32)

# Uniform numbers in the open interval (0, 1): odd multiples of 2^-24 (2^-53), which
# are exact, so neither end is reached by rounding.
@inline uniform32(w::UInt32) = (Float32(w >> 9) + 0.5f0) * 1.1920929f-7                  # 2^-23
@inline uniform64(hi::UInt32, lo::UInt32) =
    (Float64(((UInt64(hi) << 32) | UInt64(lo)) >> 12) + 0.5) * 2.220446049250313e-16     # 2^-52

# A standard normal number from four random words, by the Box-Muller transform.
@inline function normal(::Type{Float64}, w::NTuple{4,UInt32})
    u1 = uniform64(w[1], w[2])
    u2 = uniform64(w[3], w[4])
    return sqrt(-2 * log(u1)) * cospi(2 * u2)
end

@inline function normal(::Type{T}, w::NTuple{4,UInt32}) where {T<:AbstractFloat}
    u1 = uniform32(w[1])
    u2 = uniform32(w[2])
    return T(sqrt(-2 * log(u1)) * cospi(2 * u2))
end

# A uniform position in 1:count from two random words: the high 64 bits of the 96-bit
# product of a 64-bit number and `count`, whose bias is below count / 2^64.
@inline function uniform_slot(w1::UInt32, w2::UInt32, count::Int32)
    b = UInt64(count)
    high = UInt64(w1) * b
    low = UInt64(w2) * b
    return Int32((high + (low >> 32)) >> 32) + Int32(1)
end

@inline function parent_cell(occupied_idx, count, key, stream, j)
    w = random_words(key, stream, j, 0)
    return occupied_idx[uniform_slot(w[1], w[2], count)]
end

# ── Variation ───────────────────────────────────────────────────────────────────────

@kernel function gaussian_kernel!(X, @Const(genotypes), @Const(occupied_idx), count, σ, key)
    i, j = @index(Global, NTuple)
    T = eltype(X)
    p = parent_cell(occupied_idx, count, key, STREAM_PARENT, j)
    z = normal(T, random_words(key, STREAM_ENTRY, j, i))
    X[i, j] = genotypes[i, p] + σ * z
end

@kernel function isoline_kernel!(X, @Const(genotypes), @Const(occupied_idx), count, σ_iso, σ_line, key)
    i, j = @index(Global, NTuple)
    T = eltype(X)
    p1 = parent_cell(occupied_idx, count, key, STREAM_PARENT, j)
    p2 = parent_cell(occupied_idx, count, key, STREAM_SECOND_PARENT, j)
    line = σ_line * normal(T, random_words(key, STREAM_LINE, j, 0))
    z = normal(T, random_words(key, STREAM_ENTRY, j, i))
    a = genotypes[i, p1]
    X[i, j] = a + σ_iso * z + line * (genotypes[i, p2] - a)
end

function QD.device_vary!(X::AbstractMatrix{T}, op::GaussianMutation, archive::DeviceCellArchive,
                         seed::UInt64) where {T}
    isempty(X) && return X
    backend = get_backend(X)
    gaussian_kernel!(backend)(X, archive.genotypes, archive.occupied_idx, Int32(archive.count), T(op.σ),
                              key_of(seed); ndrange=size(X))
    sync!(backend)
    return X
end

function QD.device_vary!(X::AbstractMatrix{T}, op::IsoLineVariation, archive::DeviceCellArchive,
                         seed::UInt64) where {T}
    isempty(X) && return X
    backend = get_backend(X)
    isoline_kernel!(backend)(X, archive.genotypes, archive.occupied_idx, Int32(archive.count), T(op.σ_iso),
                             T(op.σ_line), key_of(seed); ndrange=size(X))
    sync!(backend)
    return X
end

# ── Evaluation ──────────────────────────────────────────────────────────────────────

# What a kernel needs of a problem: for a `QDProblem`, its function and descriptor space,
# without the generator of initial solutions, which need not be a device type.
kernel_problem(prob) = prob
kernel_problem(prob::QDProblem) = QDProblem(prob.f, nothing, prob.space)

@inline write_descriptor!(D, ::Tuple{}, j, r) = nothing
@inline function write_descriptor!(D, d::Tuple, j, r)
    D[r, j] = first(d)
    write_descriptor!(D, Base.tail(d), j, r + 1)
end

@kernel function evaluate_kernel!(fitness, descriptors, prob, @Const(X))
    j = @index(Global)
    ev = QD.device_evaluate(prob, view(X, :, j))
    fitness[j] = ev.fitness
    write_descriptor!(descriptors, Tuple(ev.descriptor), j, 1)
end

function QD.device_evaluate!(fitness::AbstractVector, descriptors::AbstractMatrix, prob::AbstractQDProblem,
                             X::AbstractMatrix)
    isempty(fitness) && return nothing
    backend = get_backend(X)
    p = kernel_problem(prob)
    backend isa KA.GPU && !isbits(p) &&
        throw(ArgumentError("$(nameof(typeof(prob))) is not a bits type, so a GPU kernel cannot take it; define QualityDiversity.evaluate_batch(prob::$(nameof(typeof(prob))), xs::DeviceBatch)"))
    evaluate_kernel!(backend)(fitness, descriptors, p, X; ndrange=length(fitness))
    sync!(backend)
    return nothing
end

# ── Insertion ───────────────────────────────────────────────────────────────────────

# An unsigned key that orders as the fitness values do.
@inline fitness_key(f::Float64) =
    (u = reinterpret(UInt64, f); signbit(f) ? ~u : u | 0x8000000000000000)
@inline fitness_key(f::Float32) =
    (u = reinterpret(UInt32, f); UInt64(signbit(f) ? ~u : u | 0x80000000))
@inline fitness_key(f::AbstractFloat) = fitness_key(Float32(f))
@inline fitness_key(f::Integer) = (Int64(f) % UInt64) ⊻ 0x8000000000000000

# The cell of column `k` of `D` on the grid, as `grid_cell` computes it on the host, or
# `0` for a descriptor that `RejectOutside()` discards and `-1` for one that the
# archive refuses.
@inline function device_cell(dims::NTuple{N,Int}, lower, upper, outside, D, k) where {N}
    index = 0
    stride = 1
    for r in 1:N
        lo = lower[r]
        hi = upper[r]
        x = Float64(D[r, k])
        if !(lo <= x <= hi)
            outside isa RejectOutside && return Int32(0)
            (outside isa ThrowOutside || isnan(x)) && return Int32(-1)
            x = clamp(x, lo, hi)
        end
        n = dims[r]
        c = min(unsafe_trunc(Int, floor((x - lo) / (hi - lo) * n)) + 1, n)
        index += (c - 1) * stride
        stride *= n
    end
    return Int32(index + 1)
end

# Whether column `k` of `D` is inside the box (`1`), rejected (`0`) or refused (`-1`) by the
# archive's `outside` policy, as `device_cell` decides it.
@inline function nearest_code(D, k, lower::NTuple{N,Float64}, upper::NTuple{N,Float64}, outside) where {N}
    code = Int32(1)
    for r in 1:N
        x = Float64(D[r, k])
        if !(lower[r] <= x <= upper[r])
            if outside isa RejectOutside
                code = min(code, Int32(0))
            elseif outside isa ThrowOutside || isnan(x)
                code = Int32(-1)
            end
        end
    end
    return code
end

# Column `k` of `D` in Float64, clamped to the box.
@inline function clamped_point(D, k, lower::NTuple{N,Float64}, upper::NTuple{N,Float64}) where {N}
    return ntuple(r -> clamp(Float64(D[r, k]), lower[r], upper[r]), Val(N))
end

# The cell of each column `k` of `D` with the nearest centroid, as the k-d tree of a host
# `CVTArchive` finds it: the same Float64 distances, summed in the same order, and the
# lowest index on a tie; `0` and `-1` as for `device_cell`. One thread per solution, which
# compares the descriptor with the centroids of its bucket only (see `CentroidBuckets`);
# they are in increasing order, so the first of equally near centroids is kept.
@kernel function nearest_kernel!(cells, @Const(centroids), @Const(offsets), @Const(members), @Const(D),
                                 bucket_dims::NTuple{N,Int}, lower::NTuple{N,Float64}, upper::NTuple{N,Float64},
                                 outside) where {N}
    k = @index(Global)
    code = nearest_code(D, k, lower, upper, outside)
    cells[k] = code == 1 ? nearest_in_bucket(centroids, offsets, members, clamped_point(D, k, lower, upper),
                                             bucket_dims, lower, upper) : code
end

@inline function nearest_in_bucket(centroids, offsets, members, q::NTuple{N,Float64}, bucket_dims::NTuple{N,Int},
                                   lower::NTuple{N,Float64}, upper::NTuple{N,Float64}) where {N}
    b = 0
    stride = 1
    for r in 1:N
        n = bucket_dims[r]
        i = min(unsafe_trunc(Int, floor((q[r] - lower[r]) / (upper[r] - lower[r]) * n)) + 1, n)
        b += (i - 1) * stride
        stride *= n
    end
    best = Int32(0)
    best_distance = Inf
    for m in offsets[b+1]:(offsets[b+2]-1)
        c = members[m]
        distance = 0.0
        for r in 1:N
            delta = centroids[r, c] - q[r]
            distance += delta * delta
        end
        if distance < best_distance
            best_distance = distance
            best = c
        end
    end
    return best
end

# Classifies solution `k`, whose cell is `c`, against the archive before the batch, and
# raises the key of the cell it enters. `flags` collects the first solution with a refused
# descriptor, the first with a NaN fitness, and whether a cell is new.
@inline function classify_solution!(k, c, cells, status, improvement, best_key, flags, occupied, stored,
                                    batch_fitness)
    F = eltype(stored)
    f = convert(F, batch_fitness[k])
    cells[k] = c
    isnan(f) && @atomic flags[2] min Int32(k)
    if c < 0
        @atomic flags[1] min Int32(k)
        status[k] = REJECTED
        improvement[k] = zero(F)
    elseif c == 0
        status[k] = REJECTED
        improvement[k] = zero(F)
    elseif occupied[c]
        gain = f - stored[c]
        improvement[k] = gain
        if gain > 0
            status[k] = IMPROVED
            @atomic best_key[c] max fitness_key(f)
        else
            status[k] = REJECTED
        end
    else
        status[k] = NEW_CELL
        improvement[k] = f
        @atomic best_key[c] max fitness_key(f)
        @atomic flags[3] max Int32(1)
    end
    return nothing
end

@kernel function grid_classify_kernel!(cells, status, improvement, best_key, flags, @Const(occupied),
                                       @Const(stored), @Const(batch_fitness), @Const(D), dims, lower, upper,
                                       outside)
    k = @index(Global)
    c = device_cell(dims, lower, upper, outside, D, k)
    classify_solution!(k, c, cells, status, improvement, best_key, flags, occupied, stored, batch_fitness)
end

# For cells found beforehand, by `nearest_kernel!`.
@kernel function found_classify_kernel!(cells, status, improvement, best_key, flags, @Const(occupied),
                                        @Const(stored), @Const(batch_fitness))
    k = @index(Global)
    classify_solution!(k, cells[k], cells, status, improvement, best_key, flags, occupied, stored, batch_fitness)
end

function classify!(backend, archive::DeviceGridArchive, cells, status, improvement, batch_fitness, D)
    grid_classify_kernel!(backend)(cells, status, improvement, archive.best_key, archive.flags, archive.occupied,
                                   archive.fitness, batch_fitness, D, archive.dims, Tuple(archive.lower),
                                   Tuple(archive.upper), archive.outside; ndrange=length(cells))
    return nothing
end

function classify!(backend, archive::DeviceCVTArchive, cells, status, improvement, batch_fitness, D)
    nearest_kernel!(backend)(cells, archive.centroid_matrix, archive.bucket_offsets, archive.bucket_members, D,
                             archive.bucket_dims, Tuple(archive.lower), Tuple(archive.upper), archive.outside;
                             ndrange=length(cells))
    sync!(backend)
    found_classify_kernel!(backend)(cells, status, improvement, archive.best_key, archive.flags, archive.occupied,
                                    archive.fitness, batch_fitness; ndrange=length(cells))
    return nothing
end

# The first solution of the batch with the best key of its cell.
@kernel function winner_kernel!(winner, @Const(best_key), @Const(cells), @Const(status), @Const(batch_fitness), F)
    k = @index(Global)
    if status[k] != REJECTED
        c = cells[k]
        fitness_key(convert(F, batch_fitness[k])) == best_key[c] && @atomic winner[c] min Int32(k)
    end
end

@kernel function place_kernel!(genotypes, stored, descriptors, occupied, @Const(winner), @Const(cells),
                               @Const(status), @Const(X), @Const(batch_fitness), @Const(D))
    i, k = @index(Global, NTuple)
    if status[k] != REJECTED
        c = cells[k]
        if winner[c] == k
            genotypes[i, c] = X[i, k]
            if i == 1
                stored[c] = batch_fitness[k]
                for r in axes(descriptors, 1)
                    descriptors[r, c] = D[r, k]
                end
                occupied[c] = true
            end
        end
    end
end

@kernel function reset_kernel!(best_key, winner, flags, @Const(cells), @Const(status))
    k = @index(Global)
    if k == 1
        flags[1] = NONE
        flags[2] = NONE
        flags[3] = Int32(0)
    end
    if status[k] != REJECTED
        c = cells[k]
        best_key[c] = zero(UInt64)
        winner[c] = NONE
    end
end

function QD.device_add_batch!(archive::DeviceCellArchive{T,F,N}, X::AbstractMatrix, batch_fitness::AbstractVector,
                              D::AbstractMatrix) where {T,F,N}
    n = length(batch_fitness)
    size(X, 2) == n ||
        throw(ArgumentError("got $(size(X, 2)) solutions but $n evaluations"))
    size(X, 1) == QD.solution_length(archive) ||
        throw(ArgumentError("solutions have $(size(X, 1)) entries but the archive stores solutions of length $(QD.solution_length(archive))"))
    size(D, 1) == N ||
        throw(ArgumentError("descriptors have $(size(D, 1)) dimensions but the archive has $N"))
    status = similar(archive.occupied_idx, UInt8, n)
    improvement = similar(archive.fitness, F, n)
    results = AddResultBatch{F,typeof(status),typeof(improvement)}(status, improvement)
    n == 0 && return results
    backend = get_backend(archive.genotypes)
    cells = similar(archive.occupied_idx, Int32, n)
    classify!(backend, archive, cells, status, improvement, batch_fitness, D)
    sync!(backend)
    refused, nan, new_cell = Array(archive.flags)
    if refused != NONE || nan != NONE
        reset_kernel!(backend)(archive.best_key, archive.winner, archive.flags, cells, status; ndrange=n)
        sync!(backend)
        if refused != NONE
            # the host computes the same cell and throws its message
            QD.cell_index(archive, SVector{N,Float64}(Array(view(D, :, Int(refused)))))
            throw(ArgumentError("the descriptor of solution $refused of the batch is outside the archive bounds"))
        end
        throw(ArgumentError("the fitness of solution $nan of the batch is NaN"))
    end
    winner_kernel!(backend)(archive.winner, archive.best_key, cells, status, batch_fitness, F; ndrange=n)
    sync!(backend)
    place_kernel!(backend)(archive.genotypes, archive.fitness, archive.descriptors, archive.occupied, archive.winner,
                           cells, status, X, batch_fitness, D; ndrange=size(X))
    sync!(backend)
    reset_kernel!(backend)(archive.best_key, archive.winner, archive.flags, cells, status; ndrange=n)
    sync!(backend)
    new_cell == 0 || QD.refresh_occupied!(archive)
    return results
end

end
