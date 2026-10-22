"""
    QualityDiversityMakieExt

Makie support for archives. `heatmap(archive)`, `heatmap!(ax, archive)` and
`plot(archive)` draw an archive with two descriptor dimensions as a heatmap
of the elite fitness, with the descriptor dimensions on the axes and the empty
cells left transparent.

- [`GridArchive`](@ref QualityDiversity.GridArchive): one rectangle per cell.
- [`SlidingBoundariesArchive`](@ref QualityDiversity.SlidingBoundariesArchive): one
  rectangle per cell, between the current
  [`boundaries`](@ref QualityDiversity.boundaries).
- [`CVTArchive`](@ref QualityDiversity.CVTArchive): the Voronoi cells, sampled on a
  `pixels` by `pixels` grid (200 by default; pass `pixels=...`).
- [`CategoricalArchive`](@ref QualityDiversity.CategoricalArchive): one square per
  combination of categories, on categorical axes that are labelled with the category
  values. Makie sorts the values of each axis, so they need `isless`.
- [`ParetoGridArchive`](@ref QualityDiversity.ParetoGridArchive): the hypervolume of the
  front of each cell.
- [`ThresholdArchive`](@ref QualityDiversity.ThresholdArchive): the result archive, as
  `solve` returns it.
- [`SpreadArchive`](@ref QualityDiversity.SpreadArchive): the archive it wraps.
- [`DensityArchive`](@ref QualityDiversity.DensityArchive): its result archive.

`scatter(archive)`, `scatter!(ax, archive)` and `plot(archive)` draw an
[`UnstructuredArchive`](@ref QualityDiversity.UnstructuredArchive) with two descriptor
dimensions as one point per stored solution, coloured by its fitness, so that
`Colorbar(fig[1, 2], plot)` shows the scale. The plot holds the scatter as its one child
and takes the attributes of `scatter`, such as `markersize`.

`ecdfplot(archive)` draws the empirical distribution function of the elite fitness of
any archive with a single real fitness, as pyribs's `archive_ecdf`: the fraction of
elites with fitness at most each value. An archive that wraps a result archive, such as a
[`ThresholdArchive`](@ref QualityDiversity.ThresholdArchive), shows its result archive.

`lines(result)` for a [`CQDScore`](@ref QualityDiversity.CQDScore) plots the mean of `ω`
over the targets against the penalty weight, the figure of Kent et al. (2022).

An archive with another number of descriptor dimensions throws an `ArgumentError`.
"""
module QualityDiversityMakieExt

using Makie: Makie, ECDFPlot, Heatmap, Lines, Scatter
using QualityDiversity: QualityDiversity, AbstractArchive, CellArchive, CQDScore, GridArchive,
    SlidingBoundariesArchive, CVTArchive, CategoricalArchive, ParetoGridArchive, ThresholdArchive,
    SpreadArchive, DensityArchive, UnstructuredArchive
using StaticArrays: SVector

const DEFAULT_PIXELS = 200

Makie.plottype(::GridArchive) = Heatmap
Makie.plottype(::SlidingBoundariesArchive) = Heatmap
Makie.plottype(::CVTArchive) = Heatmap
Makie.plottype(::CategoricalArchive) = Heatmap
Makie.plottype(::ParetoGridArchive) = Heatmap
Makie.plottype(::ThresholdArchive) = Heatmap
Makie.plottype(::SpreadArchive) = Heatmap
Makie.plottype(::DensityArchive) = Heatmap
Makie.plottype(::UnstructuredArchive) = Scatter
Makie.plottype(::CQDScore) = Lines

function require_two_dimensions(archive, N, plot="a heatmap")
    N == 2 || throw(ArgumentError(
        "$plot needs an archive with 2 descriptor dimensions but this $(nameof(typeof(archive))) has $N"))
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

function Makie.convert_arguments(::Type{<:Heatmap}, archive::SlidingBoundariesArchive{G,F,N}) where {G,F,N}
    require_two_dimensions(archive, N)
    # The boundaries are the cell edges: one more than the cells along each dimension,
    # which Makie takes as edges rather than centers.
    xedges, yedges = QualityDiversity.boundaries(archive)
    return (xedges, yedges, reshape(cell_fitness(archive), archive.dims))
end

function Makie.convert_arguments(::Type{<:Heatmap}, archive::ParetoGridArchive{G,M,N}) where {G,M,N}
    require_two_dimensions(archive, N)
    values = [archive.counts[i] > 0 ? archive.volumes[i] : NaN for i in eachindex(archive.counts)]
    return (edges(archive.lower[1], archive.upper[1], archive.dims[1]),
            edges(archive.lower[2], archive.upper[2], archive.dims[2]),
            reshape(values, archive.dims))
end

function Makie.convert_arguments(::Type{<:Heatmap}, archive::CategoricalArchive)
    require_two_dimensions(archive, length(archive.categories))
    xs, ys = archive.categories
    # Makie places the values of a categorical axis in sorted order, and a heatmap
    # needs its rows and columns in the order of their positions.
    px, py = sorted_order(xs, 1), sorted_order(ys, 2)
    values = reshape(cell_fitness(archive), length(xs), length(ys))[px, py]
    return (Makie.Categorical(xs[px]), Makie.Categorical(ys[py]), values)
end

function sorted_order(values, i)
    applicable(isless, first(values), first(values)) || throw(ArgumentError(
        "a heatmap of a CategoricalArchive sorts the categories of each axis, but categories[$i] " *
        "has values of type $(eltype(values)), which define no isless"))
    return sortperm(values)
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

Makie.used_attributes(::Type{P}, archive::SpreadArchive) where {P<:Heatmap} =
    Makie.used_attributes(P, archive.inner)

function Makie.convert_arguments(P::Type{<:Heatmap}, archive::SpreadArchive; kwargs...)
    return Makie.convert_arguments(P, archive.inner; kwargs...)
end

Makie.used_attributes(::Type{P}, archive::DensityArchive) where {P<:Heatmap} =
    Makie.used_attributes(P, QualityDiversity.result_archive(archive))

function Makie.convert_arguments(P::Type{<:Heatmap}, archive::DensityArchive; kwargs...)
    return Makie.convert_arguments(P, QualityDiversity.result_archive(archive); kwargs...)
end

function Makie.convert_arguments(::Type{<:Scatter}, archive::UnstructuredArchive{G,F,N}) where {G,F,N}
    require_two_dimensions(archive, N, "a scatter plot")
    n = length(archive)
    xs = [archive.descriptors[i][1] for i in 1:n]
    ys = [archive.descriptors[i][2] for i in 1:n]
    # A plot spec, since the colour is an attribute that converted arguments cannot carry.
    return Makie.SpecApi.Scatter(xs, ys; color=Float64.(view(archive.fitness, 1:n)))
end

function Makie.convert_arguments(P::Type{<:ECDFPlot}, archive::AbstractArchive)
    return Makie.convert_arguments(P, elite_fitness(QualityDiversity.result_archive(archive)))
end

function elite_fitness(archive::AbstractArchive{G,F}) where {G,F}
    F <: Real || throw(ArgumentError(
        "an ECDF needs a single real fitness, but this $(nameof(typeof(archive))) has fitness of type $F"))
    return Float64[ev.fitness for (_, ev) in QualityDiversity.elites(archive)]
end

Makie.convert_arguments(::Type{<:Lines}, result::CQDScore) = (result.penalties, result.per_penalty)

end
