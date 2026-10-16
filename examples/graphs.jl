# # Quality diversity over graphs
#
# The genotype here is a `Graphs.SimpleGraph`: a variable-size, discrete object
# that is not a vector of numbers. All the package needs from the user is
#
# 1. a function that scores a graph and describes it (`QDProblem`), and
# 2. a variation operator for graphs (`GraphMutation` and one `vary` method).
#
# Everything else, the archive, the emitter and the loop, is the code that also
# runs the Rastrigin example.
#
# Set up once, from the repository root:
#
#     julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run:
#
#     julia --project=examples examples/graphs.jl
#
# ## The question
#
# Which networks are efficient at carrying traffic, for each combination of
# how many edges they have and how clustered they are? Fitness is the global
# efficiency of a graph, the mean of 1/distance over all pairs of vertices
# (0 for pairs that are not connected). The descriptor has two statistics, both
# between 0 and 1: the edge density and the transitivity (the fraction of
# connected triples that close into a triangle). Efficiency rises with density,
# so the interesting comparison is between cells of equal density: how much
# efficiency does each degree of clustering cost?

module GraphExample #src

using Graphs: SimpleGraph, add_edge!, add_vertex!, edges, erdos_renyi, global_clustering_coefficient,
              gdistances, has_edge, ne, nv, rem_edge!, rem_vertex!
using QualityDiversity
using Random: AbstractRNG, Xoshiro

# ## Scoring and describing a graph

"Mean of `1/distance` over all ordered pairs of distinct vertices; unreachable pairs count 0."
function global_efficiency(g::SimpleGraph)
    n = nv(g)
    total = 0.0
    for source in 1:n
        for distance in gdistances(g, source)
            if distance != 0 && distance != typemax(Int)
                total += 1 / distance
            end
        end
    end
    return total / (n * (n - 1))
end

"Fraction of the possible edges that are present."
density(g::SimpleGraph) = 2 * ne(g) / (nv(g) * (nv(g) - 1))

"`(fitness, descriptor)` of a graph."
score(g::SimpleGraph) = (global_efficiency(g), (density(g), global_clustering_coefficient(g)))

"A problem over graphs with between `min_nodes` and `max_nodes` vertices."
function graph_problem(; min_nodes=8, max_nodes=30)
    random_graph(rng::AbstractRNG) = erdos_renyi(rand(rng, min_nodes:max_nodes), 0.2 + 0.3 * rand(rng); rng)
    return QDProblem(score; init=random_graph, bounds=((0, 1), (0, 1)))
end

# ## Varying a graph
#
# `vary` returns a changed copy and leaves the parent alone. A child gets one
# change and, with probability 1/2 each, another, and so on. A change adds an
# edge, removes one, moves one end of an edge, adds a vertex joined to a few
# others, or removes a vertex. The vertex count stays within the given limits.

"""
    GraphMutation(; min_nodes=8, max_nodes=30)

Random local changes to a `SimpleGraph`; see [`vary`](@ref QualityDiversity.vary).
"""
struct GraphMutation <: AbstractVariation
    min_nodes::Int
    max_nodes::Int
    function GraphMutation(; min_nodes=8, max_nodes=30)
        2 <= min_nodes <= max_nodes ||
            throw(ArgumentError("need 2 <= min_nodes <= max_nodes, got $min_nodes and $max_nodes"))
        return new(min_nodes, max_nodes)
    end
end

function QualityDiversity.vary(rng::AbstractRNG, op::GraphMutation, parent::SimpleGraph{Int})
    child = copy(parent)
    mutate!(rng, op, child)
    while rand(rng) < 0.5
        mutate!(rng, op, child)
    end
    return child
end

function mutate!(rng::AbstractRNG, op::GraphMutation, g::SimpleGraph{Int})
    r = rand(rng)
    if r < 0.3
        add_random_edge!(rng, g)
    elseif r < 0.6
        remove_random_edge!(rng, g)
    elseif r < 0.8
        remove_random_edge!(rng, g) && add_random_edge!(rng, g)
    elseif r < 0.9
        if nv(g) < op.max_nodes
            add_vertex!(g)
            for _ in 1:rand(rng, 1:3)
                add_edge!(g, nv(g), rand(rng, 1:nv(g)-1))
            end
        end
    elseif nv(g) > op.min_nodes
        rem_vertex!(g, rand(rng, 1:nv(g)))
    end
    return g
end

# Tries a few random pairs and adds the first missing edge; a nearly complete
# graph may stay unchanged.
function add_random_edge!(rng::AbstractRNG, g::SimpleGraph{Int})
    for _ in 1:10
        a, b = rand(rng, 1:nv(g)), rand(rng, 1:nv(g))
        if a != b && !has_edge(g, a, b)
            return add_edge!(g, a, b)
        end
    end
    return false
end

function remove_random_edge!(rng::AbstractRNG, g::SimpleGraph{Int})
    ne(g) == 0 && return false
    e = rand(rng, collect(edges(g)))
    return rem_edge!(g, e.src, e.dst)
end

# ## Running it

function main()
    prob = graph_problem()
    alg = MAPElites(prob; dims=(20, 20), batch_size=100, genotype=SimpleGraph{Int},
                    variation=GraphMutation())
    result = solve(prob, alg; budget=100_000, rng=Xoshiro(1))
    archive = result.archive

    println("evaluations: ", result.n_evals)
    println(length(archive), " of ", QualityDiversity.capacity(archive), " cells filled (coverage ",
            round(coverage(archive); digits=3), ")")
    println("best efficiency: ", round(max_fitness(archive); digits=3))

    sizes = [nv(g) for (g, _) in QualityDiversity.elites(archive)]
    println("vertices in the archive: ", minimum(sizes), " to ", maximum(sizes))

    ## At one density, the most efficient graph found for each level of clustering.
    println("\ndensity 0.25–0.35:   transitivity   vertices  edges  efficiency")
    best = Dict{Int,Any}()
    for (g, ev) in QualityDiversity.elites(archive)
        0.25 <= ev.descriptor[1] < 0.35 || continue
        bin = floor(Int, ev.descriptor[2] * 5) + 1
        if !haskey(best, bin) || ev.fitness > best[bin][2].fitness
            best[bin] = (g, ev)
        end
    end
    for bin in sort!(collect(keys(best)))
        g, ev = best[bin]
        println(rpad("", 21), rpad("$((bin - 1) / 5)–$(bin / 5)", 14), rpad(nv(g), 10), rpad(ne(g), 7),
                round(ev.fitness; digits=3))
    end

    ## A text picture of the grid: darker means more efficient, blank is empty.
    ## Columns are the density, rows the transitivity.
    shades = collect(" ░▒▓█")
    efficiencies = fill(NaN, 20, 20)
    for (_, ev) in QualityDiversity.elites(archive)
        i = QualityDiversity.cell_index(archive, ev.descriptor) - 1
        efficiencies[i ÷ 20 + 1, i % 20 + 1] = ev.fitness
    end
    lo, hi = extrema(filter(!isnan, efficiencies))
    println()
    for row in 1:20
        println(join(isnan(f) ? ' ' : shades[2 + round(Int, 3 * (f - lo) / (hi - lo))]
                     for f in efficiencies[row, :]))
    end
    return result
end

end # module #src

abspath(PROGRAM_FILE) == (@__FILE__) && GraphExample.main() #src
#md main()
