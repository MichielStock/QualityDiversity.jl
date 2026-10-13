"""
    Evaluation{F,D}(fitness, descriptor)

The result of evaluating one solution.

`fitness` is a `Real` for a single objective or an `AbstractVector` for
several. Fitness is maximized; a problem that minimizes returns the negated
value. `descriptor` locates the solution in descriptor space and is usually an
`SVector{N,Float64}`.
"""
struct Evaluation{F,D}
    fitness::F
    descriptor::D
end

"""
    AddStatus

Outcome of inserting a solution into an archive.

- `Rejected`: the archive left unchanged.
- `Improved`: the solution replaced a worse occupant of its cell.
- `NewCell`: the solution filled a previously empty cell.
"""
@enum AddStatus Rejected Improved NewCell

"""
    AddResult{F}(status, improvement)

Result of [`add!`](@ref QualityDiversity.add!).

`improvement` is the fitness gain over the previous occupant of the cell, or
over the archive's threshold for a cell that was empty. It is zero or
negative for a rejected solution. Emitters that rank
candidates by improvement, such as CMA-ME, read this field.
"""
struct AddResult{F}
    status::AddStatus
    improvement::F
end
