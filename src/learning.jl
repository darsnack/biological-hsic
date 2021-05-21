struct GlobalError{T, S, R}
    γ::T
    α::S
    ξ::R
end
GlobalError{T}(bs, n) where {T} =
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

struct HSICIdeal{T, S, R, E<:GlobalError}
    Xs::T
    Ys::T
    Zpres::T
    Zposts::T
    Kx::S
    Ky::S
    Kz::S
    λ::R
    ξ::E
end

_zero(::Type{T}, dims...) where {T<:AbstractArray} = fill!(similar(T, dims...), 0)

function HSICIdeal(xencoder::RateEncoder,
              yencoder::RateEncoder,
              layer::LIFDense{T},
              ::LIFDenseState{ZT},
              bs; λ, nbuffer) where {T, ZT}
    n = nout(layer)
    XT = eltype(xencoder)
    YT = eltype(yencoder)

    Xs = [fill!(CircularBuffer{XT}(nbuffer), _zero(XT, nout(xencoder))) for _ in 1:bs]
    Ys = [fill!(CircularBuffer{YT}(nbuffer), _zero(YT, nout(yencoder))) for _ in 1:bs]
    Zpres = [fill!(CircularBuffer{ZT}(nbuffer), _zero(ZT, nin(layer))) for _ in 1:bs]
    Zposts = [fill!(CircularBuffer{ZT}(nbuffer), _zero(ZT, n)) for _ in 1:bs]

    Kx = zeros(eltype(XT), bs, bs)
    Ky = zeros(eltype(YT), bs, bs)
    Kz = zeros(eltype(ZT), bs, bs)

    ξ = GlobalError{T}(bs, n)

    return HSICIdeal(Xs, Ys, Zpres, Zposts, Kx, Ky, Kz, λ, ξ)
end

function _shiftbuffers!(buffers, sample)
    next = popfirst!.(buffers[1:(end - 1)])
    push!(buffers[1], sample)
    push!.(buffers[2:end], next)

    return buffers
end

function (hsic::HSICIdeal)(x, y, zpre, zpost)
    # push new samples into buffers
    _shiftbuffers!(hsic.Xs, x)
    _shiftbuffers!(hsic.Ys, y)
    _shiftbuffers!(hsic.Zpres, zpre)
    _shiftbuffers!(hsic.Zposts, zpost)

    # compute new kernel matrices
    xs = reduce(hcat, getindex.(hsic.Xs, 1))
    ys = reduce(hcat, getindex.(hsic.Ys, 1))
    zposts = reduce(hcat, getindex.(hsic.Zposts, 1))
    σ = max(1f-3, estσ(zposts))
    k_hsic!(hsic.Kx, xs, xs; σ = max(1f-3, estσ(xs)))
    k_hsic!(hsic.Ky, ys, ys; σ = 0.5)
    k_hsic!(hsic.Kz, zposts, zposts; σ = σ)

    # compute local terms
    zpres = reduce(hcat, getindex.(hsic.Zpres, 1))
    @cast α[p, q, n, m] := -(hsic.Kz[p, q] / σ^2) * (zposts[n, p] - zposts[n, q]) *
        ((1 - zposts[n, p]^2) * zpres[m, p] - (1 - zposts[n, q]^2) * zpres[m, q])

    # compute weight update
    @reduce Δw[i, j] := sum(p, q) ((hsic.Kx[p, q] - @reduce _[p] := mean(n) hsic.Kx[p, n]) -
                                    hsic.λ * (hsic.Ky[p, q] - @reduce _[p] := mean(n) hsic.Ky[p, n])) *
                                  (α[p, q, i, j] - @reduce _[p, i, j] := mean(n) α[p, n, i, j])

    return -Δw
end
