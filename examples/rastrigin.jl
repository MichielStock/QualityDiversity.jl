# Fill a 20×20 grid archive on the 10-dimensional Rastrigin benchmark.
# Run from the repository root: julia --project examples/rastrigin.jl
using QualityDiversity
using Random: Xoshiro

prob = Rastrigin(10)
alg = MAPElites(prob; dims=(20, 20), batch_size=100, mutation=GaussianMutation(0.3))
result = solve(prob, alg; budget=200_000, rng=Xoshiro(1))

final = result.history[end]
println("evaluations: ", result.n_evals)
println("coverage:    ", round(final.coverage; digits=3))
println("max fitness: ", round(final.max_fitness; digits=3))
println("QD score:    ", round(final.qd_score; digits=1))
