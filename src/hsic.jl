dist(x::AbstractVector, y::AbstractVector) = sqeuclidean(x, y)
dist(xs::AbstractMatrix, ys::AbstractMatrix) = pairwise(SqEuclidean(), xs, ys; dims = 2)
dist(xs::AbstractMatrix) = pairwise(SqEuclidean(), xs; dims = 2)
dist!(zs::AbstractMatrix, xs::AbstractMatrix, ys::AbstractMatrix) =
    pairwise!(zs, SqEuclidean(), xs, ys; dims = 2)
dist!(zs::AbstractMatrix, xs::AbstractMatrix) =
    pairwise!(zs, SqEuclidean(), xs; dims = 2)

estσ(xs, ys) = Zygote.ignore() do
    ϵ = convert(eltype(xs), 1e-2)
    d = filter(!iszero, triu!(dist(xs, ys), 1))
    isempty(d) && return ϵ
    σ = mean(d)

    return max(σ, ϵ)
end
estσ(xs) = estσ(xs, xs)

rbf(d, σ) = exp.(-d ./ (2 * σ^2))
function rbf!(K, d, σ)
    K .= exp.(-d ./ (2 * σ^2))

    return K
end

k_hsic(xs, ys; σ) = rbf(dist(xs, ys), σ)
k_hsic(xs; σ) = rbf(dist(xs), σ^2)
k_hsic!(K::AbstractMatrix, xs::AbstractMatrix, ys::AbstractMatrix; σ) = rbf!(K, dist!(K, xs, ys), σ)
k_hsic!(K::AbstractMatrix, xs::AbstractMatrix; σ) = rbf!(K, dist!(K, xs), σ)

hsic(Kx, Ky, H) = tr(Kx * H * Ky * H) / (size(Kx, 1) - 1)^2

function hsic(X, Y; σx = estσ(X), σy = estσ(Y))
    N = size(X, 2)
    (size(Y, 2) != N) && error("X and Y data matrices must be same size! (X = $(size(X)) and Y = $(size(Y))")
    Kx = k_hsic(X; σ = σx)
    Ky = k_hsic(Y; σ = σy)
    H = I - fill(1 / N, N, N)

    # Zygote.ignore() do
    #     @show mean(Kx), median(Kx), extrema(Kx)
    #     @show mean(Ky), median(Ky), extrema(Ky)
    # end

    return hsic(Kx, Ky, H)
end

hsic_objective(X, Y, Z; λ = 2, σx = estσ(X), σy = estσ(Y), σz = estσ(Z)) =
    hsic(X, Z; σx = σx, σy = σz) - λ * hsic(Y, Z; σx = σy, σy = σz)
