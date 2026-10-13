# Run the suite:  julia --threads=1 benchmark/run.jl
# The first run installs the benchmark environment. Julia 1.11 and later link the
# package through the [sources] entry of Project.toml; Julia 1.10 needs develop.
using Pkg
Pkg.activate(@__DIR__)
VERSION < v"1.11" && Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
Pkg.instantiate()

using BenchmarkTools
include(joinpath(@__DIR__, "benchmarks.jl"))

tune!(SUITE)
results = run(SUITE; verbose=false)

println("Julia ", VERSION, ", ", Threads.nthreads(), " thread(s)\n")
for (path, trial) in sort!(collect(BenchmarkTools.leaves(results)); by=first)
    println(rpad(join(path, " / "), 44), rpad(BenchmarkTools.prettytime(median(trial).time), 12),
            "allocs: ", median(trial).allocs, ", memory: ", BenchmarkTools.prettymemory(median(trial).memory))
end
