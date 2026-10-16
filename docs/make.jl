# Build the documentation: julia --project=docs docs/make.jl
using Documenter
using Literate
using QualityDiversity
using CairoMakie
CairoMakie.activate!(type="png")
using KernelAbstractions

# The Makie extension has a docstring of its own, which the API reference shows.
const QualityDiversityMakieExt = Base.get_extension(QualityDiversity, :QualityDiversityMakieExt)

const EXAMPLES = joinpath(@__DIR__, "..", "examples")
const GENERATED = joinpath(@__DIR__, "src", "tutorials")

# The tutorials are the scripts in examples/, turned into pages by Literate, so the code
# in the documentation is the code that runs.
const TUTORIALS = ["rastrigin_tutorial.jl", "graphs.jl", "noisy.jl", "mome.jl", "heatmap.jl"]

rm(GENERATED; force=true, recursive=true)
for file in TUTORIALS
    Literate.markdown(joinpath(EXAMPLES, file), GENERATED; documenter=true, credit=false)
end

makedocs(;
    sitename="QualityDiversity.jl",
    modules=[QualityDiversity, QualityDiversityMakieExt],
    authors="Ward Claerhout",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://wclaerhout.github.io/QualityDiversity.jl",
        edit_link="main",
        size_threshold_warn=200 * 1024,
        size_threshold=400 * 1024,
    ),
    checkdocs=:public,
    pages=[
        "Home" => "index.md",
        "Tutorials" => ["tutorials/$(splitext(file)[1]).md" for file in TUTORIALS],
        "Guide" => [
            "guide/concepts.md",
            "guide/algorithms.md",
            "guide/extending.md",
            "guide/gpu.md",
            "guide/comparison.md",
        ],
        "API reference" => [
            "api/problems.md",
            "api/archives.md",
            "api/variation.md",
            "api/emitters.md",
            "api/algorithms.md",
            "api/metrics.md",
            "api/device.md",
            "api/testing.md",
        ],
    ],
)

deploydocs(; repo="github.com/wclaerhout/QualityDiversity.jl.git", devbranch="main", push_preview=false)
