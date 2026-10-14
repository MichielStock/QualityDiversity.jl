"""
    CMAES(dim; sigma0, batch_size=default, lower=-Inf, upper=Inf)

Covariance matrix adaptation evolution strategy (Hansen and Ostermeier, 2001;
Hansen, 2016, "The CMA Evolution Strategy: A Tutorial"), maximizing. It keeps a
Gaussian search distribution with mean `m`, step size `σ` and covariance `C`, and
is the default optimizer inside a [`CMAEmitter`](@ref).

A generation has two calls. `sample!(rng, es)` draws `batch_size` solutions,
`x = m + σ B D z` with `z` standard normal and `C = B D² Bᵀ`, and leaves them in
`es.x` as the columns of a matrix. After the caller has ranked them, `update!(es,
order, parents)` moves the distribution toward the `parents` best solutions, the
first entries of `order`. The number of parents may differ from one generation to
the next, and none is allowed; the recombination weights `log(parents + 1/2) -
log(i)` are normalized each time.

`batch_size` defaults to `4 + floor(3 log(dim))`. With `lower` and `upper`, scalars
or vectors, a sample is drawn again until it lies inside; it is an error if that
takes more than $(RESAMPLE_LIMIT) draws, which means the distribution has left the
bounds.

`should_stop(es, keys)` reports the termination criteria of CMA-ES: a covariance
matrix with a condition number above `1e14`, a distribution that has shrunk below
`1e-11`, a non-finite state, or `keys`, the ranking keys from best to worst, being
flat: the best and the worst are within `1e-12` of each other. `reset!(es, x0)`
restarts the search at `x0`.
"""
mutable struct CMAES <: AbstractEvolutionStrategy
    const dim::Int
    const batch_size::Int
    const sigma0::Float64
    const lower::Vector{Float64}
    const upper::Vector{Float64}
    mean::Vector{Float64}
    sigma::Float64
    cov::Matrix{Float64}
    # Eigendecomposition of `cov`: `cov = basis * Diagonal(eigenvalues) * basis'`,
    # and `inv_sqrt = basis * Diagonal(eigenvalues .^ -1/2) * basis'`.
    basis::Matrix{Float64}
    eigenvalues::Vector{Float64}
    inv_sqrt::Matrix{Float64}
    path_c::Vector{Float64}
    path_sigma::Vector{Float64}
    # Number of solutions drawn so far, and when the eigendecomposition was updated.
    count_eval::Int
    eigen_eval::Int
    # The last batch: `y = (x - mean) / sigma` and `x`, one solution per column.
    y::Matrix{Float64}
    x::Matrix{Float64}
    # Scratch space for a draw.
    z::Vector{Float64}
    step::Vector{Float64}
end

function CMAES(dim::Integer; sigma0::Real, batch_size::Integer=default_batch_size(dim),
               lower=-Inf, upper=Inf)
    lo, hi = check_strategy_arguments(dim, sigma0, batch_size, lower, upper)
    es = CMAES(Int(dim), Int(batch_size), Float64(sigma0), lo, hi,
               zeros(dim), Float64(sigma0), Matrix{Float64}(I, dim, dim),
               Matrix{Float64}(I, dim, dim), ones(dim), Matrix{Float64}(I, dim, dim),
               zeros(dim), zeros(dim), 0, 0,
               zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim), zeros(dim))
    return es
end

"""
    reset!(es::CMAES, x0)

Restart the search at `x0`: the mean becomes `x0`, the step size `sigma0`, and the
covariance matrix the identity.
"""
function reset!(es::CMAES, x0::AbstractVector{<:Real})
    check_start(x0, es.dim)
    es.mean .= x0
    es.sigma = es.sigma0
    for matrix in (es.cov, es.basis, es.inv_sqrt)
        fill!(matrix, 0)
        for i in 1:es.dim
            matrix[i, i] = 1
        end
    end
    es.eigenvalues .= 1
    es.path_c .= 0
    es.path_sigma .= 0
    es.count_eval = 0
    es.eigen_eval = 0
    return es
end

"""
    sample!(rng, es::CMAES)

Draw a batch from the search distribution into `es.x`, one solution per column,
and return `es.x`.
"""
function sample!(rng::AbstractRNG, es::CMAES)
    for k in 1:es.batch_size
        draws = 0
        while true
            randn!(rng, es.z)
            for i in 1:es.dim
                es.z[i] *= sqrt(es.eigenvalues[i])
            end
            mul!(es.step, es.basis, es.z)
            inside = true
            for i in 1:es.dim
                xi = es.mean[i] + es.sigma * es.step[i]
                es.x[i, k] = xi
                inside &= es.lower[i] <= xi <= es.upper[i]
            end
            if inside
                es.y[:, k] .= es.step
                break
            end
            draws += 1
            draws > RESAMPLE_LIMIT &&
                throw(resample_error("the mean is $(es.mean) and the step size $(es.sigma)"))
        end
    end
    return es.x
end

# Recombination weights for `parents` parents, and the constants of the update.
function strategy_constants(dim, parents)
    weights = recombination_weights(parents)
    mueff = 1 / sum(abs2, weights)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3)^2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)^2 + mueff))
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    return weights, mueff, cc, cs, c1, cmu, damps
end

"""
    update!(es::CMAES, order, parents)

Adapt the distribution to the last batch. `order` ranks the solutions of the batch
from best to worst, and the first `parents` of them are recombined. With no
parent only the count of evaluations advances.
"""
function update!(es::CMAES, order::AbstractVector{<:Integer}, parents::Integer)
    length(order) == es.batch_size ||
        throw(ArgumentError("order has $(length(order)) entries but the batch has $(es.batch_size)"))
    0 <= parents <= es.batch_size ||
        throw(ArgumentError("parents must be between 0 and $(es.batch_size), got $parents"))
    es.count_eval += es.batch_size
    parents == 0 && return es
    n = es.dim
    weights, mueff, cc, cs, c1, cmu, damps = strategy_constants(n, parents)
    chi_n = sqrt(n) * (1 - 1 / (4n) + 1 / (21n^2))

    ys = es.y[:, order[1:parents]]
    yw = ys * weights                                    # mean shift in units of sigma
    es.mean .+= es.sigma .* yw

    # Evolution path of the step size, and the stall indicator for the path of C.
    es.path_sigma .= (1 - cs) .* es.path_sigma .+ sqrt(cs * (2 - cs) * mueff) .* (es.inv_sqrt * yw)
    norm_ps = norm(es.path_sigma)
    decay = 1 - (1 - cs)^(2 * es.count_eval / es.batch_size)
    hsig = norm_ps / sqrt(decay) / chi_n < 1.4 + 2 / (n + 1) ? 1.0 : 0.0
    es.path_c .= (1 - cc) .* es.path_c .+ hsig * sqrt(cc * (2 - cc) * mueff) .* yw

    # Covariance: rank-one update from the path and rank-mu update from the parents.
    delta = (1 - hsig) * cc * (2 - cc)
    es.cov .*= 1 + c1 * delta - c1 - cmu
    es.cov .+= c1 .* (es.path_c * es.path_c')
    mul!(es.cov, ys .* weights', ys', cmu, 1.0)

    es.sigma *= exp(min(1.0, (cs / damps) * (norm_ps / chi_n - 1)))

    # The decomposition is only recomputed now and then; it costs O(n^3).
    if es.count_eval - es.eigen_eval > es.batch_size / ((c1 + cmu) * n * 10)
        update_eigensystem!(es)
    end
    return es
end

function update_eigensystem!(es::CMAES)
    n = es.dim
    for j in 1:n, i in 1:j-1
        es.cov[i, j] = es.cov[j, i] = (es.cov[i, j] + es.cov[j, i]) / 2
    end
    decomposition = eigen(Symmetric(es.cov))
    if all(isfinite, decomposition.values) && all(isfinite, decomposition.vectors)
        es.eigenvalues .= max.(decomposition.values, floatmin(Float64))
        es.basis .= decomposition.vectors
        es.inv_sqrt .= es.basis * Diagonal(1 ./ sqrt.(es.eigenvalues)) * es.basis'
    else
        es.eigenvalues .= NaN
    end
    es.eigen_eval = es.count_eval
    return es
end

"""
    should_stop(es::CMAES, keys) -> Bool

Whether the search has converged or broken down; see [`CMAES`](@ref). `keys` are the
ranking keys of the last batch from best to worst.
"""
function should_stop(es::CMAES, keys::AbstractVector)
    return covariance_converged(es.sigma, es.mean, es.eigenvalues, keys)
end
