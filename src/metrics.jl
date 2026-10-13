"""
    qd_score(archive; offset)

Sum of `fitness + offset` over the occupied cells of `archive`. The offset
keeps negative fitness values from lowering the score; use
[`fitness_offset`](@ref QualityDiversity.fitness_offset) of the problem as a
starting point.
"""
function qd_score end

"""
    coverage(archive)

Fraction of the cells of `archive` that are occupied, in `[0, 1]`.
"""
function coverage end

"""
    max_fitness(archive)

Highest fitness among the occupied cells of `archive`.
"""
function max_fitness end

"""
    mean_fitness(archive)

Mean fitness over the occupied cells of `archive`.
"""
function mean_fitness end
