"""
    AbstractRanker

Supertype of the rules that order the candidates of a [`CMAEmitter`](@ref) from best
to worst after they have been evaluated and added to the archive. The optimizer
moves toward the best ones, so the ranker decides what the emitter looks for.

A ranker defines `rank_key(ranker, ev, result)`, which gives a key for one candidate
from its [`Evaluation`](@ref) and its [`AddResult`](@ref). Candidates are ordered by
descending key; ties keep their order in the batch. A key is a `Float64` or a tuple
that compares lexicographically, and
[`ranking_value`](@ref QualityDiversity.ranking_value) of it is the number that
measures the candidate, which the emitter uses to notice that a batch is flat.

The rankers are [`TwoStageImprovementRanker`](@ref), [`ImprovementRanker`](@ref) and
[`ObjectiveRanker`](@ref).
"""
abstract type AbstractRanker end

"""
    ObjectiveRanker()

Ranks candidates by fitness alone, whatever the archive did with them. A
[`CMAEmitter`](@ref) with this ranker is a plain CMA-ES that maximizes the fitness,
the optimizing emitter of CMA-ME.
"""
struct ObjectiveRanker <: AbstractRanker end

"""
    ImprovementRanker()

Ranks candidates by the `improvement` of their [`AddResult`](@ref), the gain over the
threshold of the cell. This is the ranking of CMA-MAE, for use with a
[`ThresholdArchive`](@ref).

For an archive where a new cell reports its fitness as `improvement`, as the other
cell archives do, the improvements of new and old cells are on different scales;
[`TwoStageImprovementRanker`](@ref) is the ranking for those.
"""
struct ImprovementRanker <: AbstractRanker end

"""
    TwoStageImprovementRanker()

Ranks candidates first by what the archive did with them, a new cell above an
improved cell above a rejection, and then by `improvement`. This is the ranking of the
improvement emitter of CMA-ME (Fontaine et al., 2020).
"""
struct TwoStageImprovementRanker <: AbstractRanker end

"""
    rank_key(ranker::AbstractRanker, ev::Evaluation, result::AddResult)

The key that orders one candidate: a `Float64`, or a tuple that compares
lexicographically, where a larger key ranks earlier. `ev` is the evaluation of the
candidate and `result` what the archive made of it. A new ranker defines this method.
"""
function rank_key end

rank_key(::ObjectiveRanker, ev, result) = Float64(ev.fitness)

rank_key(::ImprovementRanker, ev, result) = Float64(result.improvement)

function rank_key(::TwoStageImprovementRanker, ev, result)
    stage = result.status == NewCell ? 2 : result.status == Improved ? 1 : 0
    return (stage, Float64(result.improvement))
end

"""
    ranking_value(key)

The number in a ranking key that measures the candidate: the key itself, or the last
entry of a tuple. An emitter treats a batch whose ranking values are all equal as
flat and restarts.
"""
ranking_value(key::Float64) = key
ranking_value(key::Tuple) = Float64(last(key))

# Order of the candidates from best to worst, and their ranking values in that order.
function rank(ranker::AbstractRanker, evs, results)
    keys = [rank_key(ranker, ev, result) for (ev, result) in zip(evs, results)]
    order = sortperm(keys; rev=true, alg=MergeSort)
    values = Float64[ranking_value(keys[i]) for i in order]
    return order, values
end
