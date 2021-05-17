dist(x, y) = norm(x - y)^2

estσ(xs) = median([dist(x, x̂) for x in eachcol(xs), x̂ in eachcol(xs)])
estσ(xs, ys) = median(dist(z, ẑ) for z in eachcol(hcat(xs, ys)), ẑ in eachcol(hcat(xs, ys)))

k_hsic(x, y; σ) = exp(-dist(x, y) / (2 * σ^2))

function hsic(X, Y; σx = estσ(X), σy = estσ(Y))
    N = size(X, 2)
    (size(Y, 2) != N) && error("X and Y data matrices must be same size! (X = $(size(X)) and Y = $(size(Y))")
    Kx = [k(x, x̂; σ = σx) for x in eachcol(X), x̂ in eachcol(X)]
    Ky = [k(y, ŷ; σ = σy) for y in eachcol(Y), ŷ in eachcol(Y)]
    H = I - fill(1 / N, N, N)

    return tr(Kx * H * Ky * H) / (N - 1)^2
end