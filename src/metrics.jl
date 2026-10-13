"""
    qd_score(archive; offset=0)

Sum of `fitness + offset` over the occupied cells of `archive`. The offset
keeps negative fitness values from lowering the score; use
[`fitness_offset`](@ref QualityDiversity.fitness_offset) of the problem as a
starting point.

The method for any [`AbstractArchive`](@ref) makes one pass over
[`elites`](@ref QualityDiversity.elites). An archive that keeps running totals
specializes it, as [`GridArchive`](@ref) does.
"""
function qd_score(archive::AbstractArchive; offset=0)
    total = 0.0
    for (_, ev) in elites(archive)
        total += ev.fitness + offset
    end
    return total
end

"""
    coverage(archive)

Fraction of the cells of `archive` that are occupied, in `[0, 1]`.
"""
coverage(archive::AbstractArchive) = length(archive) / capacity(archive)

"""
    max_fitness(archive)

Highest fitness among the occupied cells of `archive`. Throws an
`ArgumentError` if the archive is empty.
"""
function max_fitness(archive::AbstractArchive)
    isempty_archive(archive, "max_fitness")
    return maximum(ev.fitness for (_, ev) in elites(archive))
end

"""
    mean_fitness(archive)

Mean fitness over the occupied cells of `archive`. Throws an `ArgumentError`
if the archive is empty.
"""
function mean_fitness(archive::AbstractArchive)
    isempty_archive(archive, "mean_fitness")
    total = 0.0
    for (_, ev) in elites(archive)
        total += ev.fitness
    end
    return total / length(archive)
end

function isempty_archive(archive, name)
    length(archive) == 0 &&
        throw(ArgumentError("$name is undefined for an archive with no elites"))
    return nothing
end
