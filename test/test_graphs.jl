# The graph example: a genotype that is a Graphs.jl object, with its own
# variation operator. The conformance checks run on it exactly as on vectors.
using Graphs: SimpleGraph, add_edge!, complete_graph, erdos_renyi, ne, nv, path_graph

include(joinpath(@__DIR__, "..", "examples", "graphs.jl"))
using .GraphExample: GraphMutation, global_efficiency, graph_problem, score
using QualityDiversity.Testing

@testset "graph statistics" begin
    @test global_efficiency(complete_graph(5)) == 1
    @test global_efficiency(SimpleGraph(5)) == 0
    @test global_efficiency(path_graph(3)) ≈ 5 / 6
    fitness, (density, transitivity) = score(complete_graph(6))
    @test (fitness, density, transitivity) == (1.0, 1.0, 1.0)
    fitness, (density, transitivity) = score(path_graph(4))
    @test density ≈ 3 / 6
    @test transitivity == 0
    g = SimpleGraph(4)
    add_edge!(g, 1, 2)
    @test global_efficiency(g) ≈ 2 / 12     # one connected pair, counted in both directions
end

@testset "GraphMutation" begin
    op = GraphMutation(min_nodes=6, max_nodes=12)
    @test_throws "need 2 <= min_nodes <= max_nodes, got 5 and 3" GraphMutation(min_nodes=5, max_nodes=3)

    rng = Xoshiro(1)
    parent = erdos_renyi(9, 0.3; rng)
    snapshot = copy(parent)
    changed = 0
    for _ in 1:2_000
        child = QD.vary(rng, op, parent)
        @test child isa SimpleGraph{Int}
        @test 6 <= nv(child) <= 12
        changed += child != parent
        parent = child
    end
    @test changed > 1_000                      # most children differ from their parent
    @test snapshot == erdos_renyi(9, 0.3; rng=Xoshiro(1))

    # the vertex count reaches both limits
    sizes = Set{Int}()
    g = erdos_renyi(9, 0.3; rng)
    for _ in 1:5_000
        g = QD.vary(rng, op, g)
        push!(sizes, nv(g))
    end
    @test 6 in sizes && 12 in sizes
end

@testset "conformance with a graph genotype" begin
    prob = graph_problem(min_nodes=6, max_nodes=14)
    op = GraphMutation(min_nodes=6, max_nodes=14)
    G = SimpleGraph{Int}

    test_problem(prob)
    test_archive(() -> GridArchive(prob; dims=(8, 8), genotype=G), prob)
    test_archive(() -> GridArchive(prob; dims=(8, 8), genotype=G, outside=ClampOutside()), prob)
    test_variation(op, erdos_renyi(10, 0.3; rng=Xoshiro(2)))

    archive = GridArchive(prob; dims=(8, 8), genotype=G)
    xs = [QD.random_solution(Xoshiro(i), prob) for i in 1:30]
    QD.add_batch!(archive, xs, QD.evaluate_batch(prob, xs))
    test_emitter(VariationEmitter(op; batch_size=20), archive, prob)
end

@testset "MAP-Elites over graphs" begin
    prob = graph_problem(min_nodes=6, max_nodes=14)
    alg = MAPElites(prob; dims=(10, 10), batch_size=50, genotype=SimpleGraph{Int},
                    mutation=GraphMutation(min_nodes=6, max_nodes=14))
    result = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test result.n_evals == 5_000
    @test result.archive isa GridArchive{SimpleGraph{Int}}
    @test length(result.archive) > 20
    @test issorted(r.qd_score for r in result.history)

    sizes = [nv(g) for (g, _) in QD.elites(result.archive)]
    @test minimum(sizes) >= 6 && maximum(sizes) <= 14
    @test length(unique(sizes)) > 3             # the archive holds graphs of several sizes

    # every stored fitness and descriptor is that of the stored graph
    for (g, ev) in QD.elites(result.archive)
        fitness, descriptor = score(g)
        @test ev.fitness == fitness
        @test ev.descriptor == SVector(descriptor...)
    end

    again = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test again.history == result.history
end
