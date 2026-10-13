# Nearest-point search over a fixed set of centroids, and k-means to choose them.
#
# `CentroidTree` is a k-d tree stored as the permutation `order` of the points:
# the point in the middle of a range is the root of that range, and the ranges
# on either side are its subtrees. Splitting along the axis of largest spread
# keeps the tree useful in the handful of dimensions descriptors have. A search
# allocates nothing.

struct CentroidTree{N}
    points::Vector{SVector{N,Float64}}   # in the order of the cells
    order::Vector{Int}                   # indices of `points`, arranged as a tree
    axis::Vector{Int8}                   # split axis of the node at each position of `order`
end

function CentroidTree(points::Vector{SVector{N,Float64}}) where {N}
    order = collect(1:length(points))
    axis = ones(Int8, length(points))
    build!(order, axis, points, 1, length(points))
    return CentroidTree{N}(points, order, axis)
end

function build!(order, axis, points::Vector{SVector{N,Float64}}, lo, hi) where {N}
    lo >= hi && return nothing
    lows = points[order[lo]]
    highs = lows
    for k in lo+1:hi
        p = points[order[k]]
        lows = min.(lows, p)
        highs = max.(highs, p)
    end
    split = argmax(highs - lows)
    sort!(view(order, lo:hi); by=i -> points[i][split])
    mid = (lo + hi) >> 1
    axis[mid] = split
    build!(order, axis, points, lo, mid - 1)
    build!(order, axis, points, mid + 1, hi)
    return nothing
end

"""
    nearest(tree::CentroidTree, q) -> Int

Index of the point of `tree` closest to `q` in Euclidean distance; the lowest
index among equally close points.
"""
function nearest(tree::CentroidTree{N}, q::SVector{N,Float64}) where {N}
    index, _ = search(tree, q, 1, length(tree.order), 0, Inf)
    return index
end

function search(tree::CentroidTree{N}, q::SVector{N,Float64}, lo::Int, hi::Int,
                best::Int, best_distance::Float64)::Tuple{Int,Float64} where {N}
    lo > hi && return (best, best_distance)
    mid = (lo + hi) >> 1
    i = tree.order[mid]
    c = tree.points[i]
    distance = sum(abs2, c - q)
    if distance < best_distance || (distance == best_distance && i < best)
        best, best_distance = i, distance
    end
    lo == hi && return (best, best_distance)
    axis = tree.axis[mid]
    offset = q[axis] - c[axis]
    if offset < 0
        best, best_distance = search(tree, q, lo, mid - 1, best, best_distance)
        if offset * offset <= best_distance
            best, best_distance = search(tree, q, mid + 1, hi, best, best_distance)
        end
    else
        best, best_distance = search(tree, q, mid + 1, hi, best, best_distance)
        if offset * offset <= best_distance
            best, best_distance = search(tree, q, lo, mid - 1, best, best_distance)
        end
    end
    return (best, best_distance)
end

# k-means with k-means++ seeding. Lloyd's iterations stop when no sample changes
# cluster, or when an iteration lowers the total squared distance from the
# samples to their centroids by less than the fraction `tolerance` of it.
function kmeans(rng::AbstractRNG, samples::Vector{SVector{N,Float64}}, k::Int;
                max_iterations::Int=100, tolerance::Float64=1e-3) where {N}
    centers = kmeans_seeds(rng, samples, k)
    labels = zeros(Int, length(samples))
    sums = zeros(SVector{N,Float64}, k)
    counts = zeros(Int, k)
    previous = Inf
    for _ in 1:max_iterations
        tree = CentroidTree(centers)
        fill!(sums, zero(SVector{N,Float64}))
        fill!(counts, 0)
        changed = false
        total = 0.0
        for (s, x) in enumerate(samples)
            c = nearest(tree, x)
            changed |= labels[s] != c
            labels[s] = c
            total += sum(abs2, x - centers[c])
            sums[c] += x
            counts[c] += 1
        end
        for j in 1:k
            # a cluster that lost all its samples restarts at a random sample
            centers[j] = counts[j] > 0 ? sums[j] / counts[j] : samples[rand(rng, 1:length(samples))]
        end
        (changed && previous - total > tolerance * total) || break
        previous = total
    end
    return centers
end

# k-means++: each seed is a sample drawn with probability proportional to its
# squared distance from the seeds chosen so far.
function kmeans_seeds(rng::AbstractRNG, samples::Vector{SVector{N,Float64}}, k::Int) where {N}
    n = length(samples)
    seeds = Vector{SVector{N,Float64}}(undef, k)
    seeds[1] = samples[rand(rng, 1:n)]
    distance = [sum(abs2, x - seeds[1]) for x in samples]
    for j in 2:k
        total = sum(distance)
        pick = n
        if total > 0
            threshold = rand(rng) * total
            running = 0.0
            for i in 1:n
                running += distance[i]
                if running >= threshold
                    pick = i
                    break
                end
            end
        else
            pick = rand(rng, 1:n)
        end
        seeds[j] = samples[pick]
        for i in 1:n
            distance[i] = min(distance[i], sum(abs2, samples[i] - seeds[j]))
        end
    end
    return seeds
end
