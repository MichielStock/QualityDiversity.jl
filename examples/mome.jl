# Multi-objective MAP-Elites on the two-objective BiSphere benchmark: every cell of a
# 20×20 grid keeps a Pareto front of solutions that trade off the two objectives.
# Run from the repository root: julia --project examples/mome.jl
using QualityDiversity
using Random: Xoshiro

prob = BiSphere(10)
alg = MOME(prob; dims=(20, 20), batch_size=64, variation=GaussianMutation(0.2))
result = solve(prob, alg; budget=100_000, rng=Xoshiro(1))
archive = result.archive

final = result.history[end]
println("evaluations:         ", result.n_evals)
println("occupied cells:      ", length(archive), " of ", QualityDiversity.capacity(archive))
println("solutions:           ", final.num_solutions)
println("MOQD score:          ", round(final.qd_score; digits=1))
println("largest cell volume: ", round(final.max_hypervolume; digits=1))
println("global hypervolume:  ", round(global_hypervolume(archive); digits=1))
println("best objectives:     ", round.(max_scores(archive); digits=3))
