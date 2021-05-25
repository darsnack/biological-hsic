dist(x::AbstractVector, y::AbstractVector) = norm(x - y)^2
dist(xs::AbstractMatrix, ys::AbstractMatrix) = @reduce _[i, j] := sum(μ) (xs[μ, i] - ys[μ, j])^2
function dist!(zs::AbstractMatrix, xs::AbstractMatrix, ys::AbstractMatrix)
    @reduce zs[i, j] := sum(μ) (xs[μ, i], ys[μ, j])^2
    return zs
end

estσ(xs) = median(dist(xs, xs))
estσ(xs, ys) = median(dist(xs, ys))

k_hsic(x::AbstractVector, y::AbstractVector; σ) = exp(-dist(x, y) / (2 * σ^2))
k_hsic(xs::AbstractMatrix, ys::AbstractMatrix; σ) =
    @cast _[i, j] := exp(-(@reduce _[i, j] := sum(μ) (xs[μ, i] - ys[μ, j])^2) / (2 * σ^2))
function k_hsic!(K::AbstractMatrix, xs::AbstractMatrix, ys::AbstractMatrix; σ)
    @cast K[i, j] = exp(-(@reduce _[i, j] := sum(μ) (xs[μ, i] - ys[μ, j])^2) / (2 * σ^2))
    return K
end

function hsic(X, Y; σx = estσ(X), σy = estσ(Y))
    N = size(X, 2)
    (size(Y, 2) != N) && error("X and Y data matrices must be same size! (X = $(size(X)) and Y = $(size(Y))")
    Kx = k_hsic(xs, xs; σ = σx)
    Ky = k_hsic(ys, ys; σ = σy)
    H = I - fill(1 / N, N, N)

    return tr(Kx * H * Ky * H) / (N - 1)^2
end

hsic_objective(X, Y, Z; λ = 2, σx = estσ(X), σy = estσ(Y), σz = estσ(Z)) =
    hsic(X, Z; σx = σx, σy = σz) - λ * hsic(Y, Z; σx = σy, σy = σz)
