# Draw the archives of a run as heatmaps and save them to examples/heatmaps.png.
# Run from the repository root: julia --project=examples examples/heatmap.jl
using QualityDiversity
using CairoMakie
using Random: Xoshiro

prob = Rastrigin(10)
bounds = QualityDiversity.descriptor_bounds(prob)
grid = solve(prob, MAPElites(prob; dims=(40, 40), batch_size=100); budget=100_000, rng=Xoshiro(1))
cvt = solve(prob, MAPElites(CVTArchive(Xoshiro(1), 1_000, bounds); batch_size=100);
            budget=100_000, rng=Xoshiro(1))

fig = Figure(size=(1000, 420))
for (column, (title, result)) in enumerate(("GridArchive" => grid, "CVTArchive" => cvt))
    ax = Axis(fig[1, 2*column-1]; title, xlabel="descriptor 1", ylabel="descriptor 2", aspect=1)
    hm = heatmap!(ax, result.archive)
    Colorbar(fig[1, 2*column], hm; label="fitness")
end
save(joinpath(@__DIR__, "heatmaps.png"), fig)
