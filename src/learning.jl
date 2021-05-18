struct GlobalError{T, S, R}
    γ::T
    α::S
    ξ::R
end
GlobalError{T}(bs, n) where T =
    GlobalError(zeros(T, bs, bs), [zeros(T, n) for _ in 1:bs, _ in 1:bs], zeros(T, n))

function (error::GlobalError)(kx, ky, kz, z)
    bs = size(error.γ, 1)

    error.γ .= (kx .- mean(kx; dims = 2)) - 2 * (ky .- mean(ky; dims = 2))

    for j in 1:bs, i in 1:bs
        error.α[i, j] = -kz[i, j] * (z[:, i] - z[:, j]) / σz^2
    end
    error.α .= error.α .- mean(error.α; dims = 2)

    error.ξ .= sum(error.γ[i, j] * error.α[i, j] for i in 1:bs, j in 1:bs) / (bs - 1)^2

    return error.ξ
end