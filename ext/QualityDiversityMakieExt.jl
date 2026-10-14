"""
    QualityDiversityMakieExt

Makie support for archives. `heatmap(archive)`, `heatmap!(ax, archive)` and
`plot(archive)` draw an archive with two descriptor dimensions as a heatmap
of the elite fitness, with the descriptor dimensions on the axes and the empty
cells left transparent.

- [`GridArchive`](@ref QualityDiversity.GridArchive): one rectangle per cell.
- [`CVTArchive`](@ref QualityDiversity.CVTArchive): the Voronoi cells, sampled on a
  `pixels` by `pixels` grid (200 by default; pass `pixels=...`).
- [`ParetoGridArchive`](@ref QualityDiversity.ParetoGridArchive): the hypervolume of the
  front of each cell.
- [`ThresholdArchive`](@ref QualityDiversity.ThresholdArchive): the result archive, as
  `solve` returns it.

An archive with another number of descriptor dimensions throws an `ArgumentError`.
"""
module QualityDiversityMakieExt

using Makie: Makie, Heatmap
using QualityDiversity: QualityDiversity, CellArchive, GridArchive, CVTArchive,
    ParetoGridArchive, ThresholdArchive
using StaticArrays: SVector

const DEFAULT_PIXELS = 200

Makie.plottype(::GridArchive) = Heatmap
Makie.plottype(::CVTArchive) = Heatmap
Makie.plottype(::ParetoGridArchive) = Heatmap
Makie.plottype(::ThresholdArchive) = Heatmap

function require_two_dimensions(archive, N)
    N == 2 || throw(ArgumentError(
        "a heatmap needs an archive with 2 descriptor dimensions but this $(nameof(typeof(archive))) has $N"))
    return nothing
end

# Fitness of the occupied cells in storage order, `NaN` for the empty ones.
function cell_fitness(archive::CellArchive)
    return [archive.occupied[i] ? Float64(archive.fitness[i]) : NaN
            for i in eachindex(archive.occupied)]
end

edges(lower, upper, n) = collect(range(lower, upper; length=n + 1))

function Makie.convert_arguments(::Type{<:Heatmap}, archive::GridArchive{G,F,N}) where {G,F,N}
    require_two_dimensions(archive, N)
    # Cells are numbered with the first dimension varying fastest, which is the
    # layout of a matrix indexed by (x, y).
    values = reshape(cell_fitness(archive), archive.dims)
    return (edges(archive.lower[1], archive.upper[1], archive.dims[1]),
            edges(archive.lower[2], archive.upper[2], archive.dims[2]),
            values)
end

function Makie.convert_arguments(::Type{<:Heatmap}, archive::ParetoGridArchive{G,M,N}) where {G,M,N}
    require_two_dimensions(archive, N)
    values = [archive.counts[i] > 0 ? archive.volumes[i] : NaN for i in eachindex(archive.counts)]
    return (edges(archive.lower[1], archive.upper[1], archive.dims[1]),
            edges(archive.lower[2], archive.upper[2], archive.dims[2]),
            reshape(values, archive.dims))
end

Makie.used_attributes(::Type{<:Heatmap}, ::CVTArchive) = (:pixels,)

function Makie.convert_arguments(::Type{<:Heatmap}, archive::CVTArchive{G,F,N};
                                 pixels::Integer=DEFAULT_PIXELS) where {G,F,N}
    require_two_dimensions(archive, N)
    pixels >= 1 || throw(ArgumentError("pixels must be at least 1, got $pixels"))
    xedges = edges(archive.lower[1], archive.upper[1], pixels)
    yedges = edges(archive.lower[2], archive.upper[2], pixels)
    fitness = cell_fitness(archive)
    values = Matrix{Float64}(undef, pixels, pixels)
    for j in 1:pixels, i in 1:pixels
        center = SVector((xedges[i] + xedges[i+1]) / 2, (yedges[j] + yedges[j+1]) / 2)
        values[i, j] = fitness[QualityDiversity.cell_index(archive, center)]
    end
    return (xedges, yedges, values)
end

Makie.used_attributes(::Type{P}, archive::ThresholdArchive) where {P<:Heatmap} =
    Makie.used_attributes(P, QualityDiversity.result_archive(archive))

function Makie.convert_arguments(P::Type{<:Heatmap}, archive::ThresholdArchive; kwargs...)
    return Makie.convert_arguments(P, QualityDiversity.result_archive(archive); kwargs...)
end

end
