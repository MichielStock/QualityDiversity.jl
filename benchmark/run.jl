# Run the suite:  julia --threads=1 benchmark/run.jl
# The first run installs the benchmark environment and links the package.
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
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
