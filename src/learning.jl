struct GlobalError{T, S, R}
    γ::T
    α::S
    ξ::R
end
GlobalError{T}(bs, n) where {T} =
    GlobalError(zeros(T, bs), zeros(T, bs, n), zeros(T, n))

Adapt.adapt_structure(to, error::GlobalError) = GlobalError(adapt(to, error.γ),
                                                            adapt(to, error.α),
                                                            adapt(to, error.ξ))
cpu(error::GlobalError) = adapt(Array, error)
gpu(error::GlobalError) = adapt(CuArray, error)

function (error::GlobalError)(kx, ky, kz, z; σz = estσ(z))
    bs = size(error.γ, 1)

    @cast error.γ[i] = (kx[1, i] - @reduce sum(k) kx[1, k] / bs) -
                        2 * (ky[1, i] - @reduce sum(k) ky[1, k] / bs)

    @cast error.α[i, k] = -2 * kz[1, i] * (z[k, 1] - z[k, i]) / σz^2
    @cast error.α[i, k] = error.α[i, k] - @reduce _[k] := sum(n) error.α[n, k]

    @reduce error.ξ[k] = sum(i) 1000 * error.γ[i] * error.α[i, k] / (bs - 1)^2

    return error.ξ
end

function _shiftbuffers!(buffers, sample)
    next = popfirst!.(buffers[1:(end - 1)])
    push!(buffers[1], sample)
    push!.(buffers[2:end], next)

    return buffers
end

struct HSICIdeal{T, S, R}
    Xs::T
    Ys::T
    Zpres::T
    Zposts::T
    Kx::S
    Ky::S
    Kz::S
    λ::R
end

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

    return HSICIdeal(Xs, Ys, Zpres, Zposts, Kx, Ky, Kz, λ)
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

struct HSICApprox{T, S, R}
    Xs::T
    Ys::T
    Zposts::T
    Kx::S
    Ky::S
    Kz::S
    λ::R
end

function HSICApprox(xencoder::RateEncoder,
              yencoder::RateEncoder,
              layer::LIFDense{T},
              ::LIFDenseState{ZT},
              bs; λ, nbuffer) where {T, ZT}
    n = nout(layer)
    XT = eltype(xencoder)
    YT = eltype(yencoder)

    Xs = [fill!(CircularBuffer{XT}(nbuffer), _zero(XT, nout(xencoder))) for _ in 1:bs]
    Ys = [fill!(CircularBuffer{YT}(nbuffer), _zero(YT, nout(yencoder))) for _ in 1:bs]
    Zposts = [fill!(CircularBuffer{ZT}(nbuffer), _zero(ZT, n)) for _ in 1:bs]

    Kx = zeros(eltype(XT), bs, bs)
    Ky = zeros(eltype(YT), bs, bs)
    Kz = zeros(eltype(ZT), bs, bs)

    return HSICApprox(Xs, Ys, Zposts, Kx, Ky, Kz, λ)
end

function (hsic::HSICApprox)(x, y, zpre, zpost)
    # push new samples into buffers
    _shiftbuffers!(hsic.Xs, x)
    _shiftbuffers!(hsic.Ys, y)
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
    @cast α[q, n, m] := -2 * (hsic.Kz[1, q] / σ^2) * (zposts[n, 1] - zposts[n, q]) *
                         ((1 - zposts[n, 1]^2) * zpre[m])

    # compute weight update
    @reduce Δw[i, j] := sum(q) ((hsic.Kx[1, q] - @reduce _[] := mean(n) hsic.Kx[1, n]) -
                                 hsic.λ * (hsic.Ky[1, q] - @reduce _[] := mean(n) hsic.Ky[1, n])) *
                                (α[q, i, j] - @reduce _[i, j] := mean(n) α[n, i, j])

    return -Δw
end

struct HSIC{T, S, R, E<:GlobalError, P<:Reservoir, Q<:RMHebb}
    Xs::T
    Ys::T
    Zposts::T
    Kx::S
    Ky::S
    Kz::S
    γ::R
    ξ::E
    reservoir::P
    learner::Q
end

function HSIC(xencoder::RateEncoder,
              yencoder::RateEncoder,
              layer::LIFDense{T},
              ::LIFDenseState{ZT},
              reservoir::Reservoir,
              learner::RMHebb,
              bs; γ, nbuffer) where {T, ZT}
    n = nout(layer)
    XT = eltype(xencoder)
    YT = eltype(yencoder)

    Xs = [fill!(CircularBuffer{XT}(nbuffer), _zero(XT, nout(xencoder))) for _ in 1:bs]
    Ys = [fill!(CircularBuffer{YT}(nbuffer), _zero(YT, nout(yencoder))) for _ in 1:bs]
    Zposts = [fill!(CircularBuffer{ZT}(nbuffer), _zero(ZT, n)) for _ in 1:bs]

    Kx = _zero(XT, bs, bs)
    Ky = _zero(YT, bs, bs)
    Kz = _zero(ZT, bs, bs)

    ξ = adapt(ZT, GlobalError{T}(bs, n))

    return HSIC(Xs, Ys, Zposts, Kx, Ky, Kz, γ, ξ, reservoir, learner)
end

state(hsic::HSIC) = state(hsic.reservoir)

function (hsic::HSIC)(state, x, y, zpre, zpost, t, Δt; explore = true)
    # push new samples into buffers
    _shiftbuffers!(hsic.Xs, x)
    _shiftbuffers!(hsic.Ys, y)
    _shiftbuffers!(hsic.Zposts, zpost)

    # compute new kernel matrices
    xs = reduce(hcat, getindex.(hsic.Xs, 1))
    ys = reduce(hcat, getindex.(hsic.Ys, 1))
    zposts = reduce(hcat, getindex.(hsic.Zposts, 1))
    σ = max(1f-3, estσ(zposts))
    k_hsic!(hsic.Kx, xs, xs; σ = max(1f-3, estσ(xs)))
    k_hsic!(hsic.Ky, ys, ys; σ = 0.5)
    k_hsic!(hsic.Kz, zposts, zposts; σ = σ)

    # compute global term
    ξ = hsic.ξ(hsic.Kx, hsic.Ky, hsic.Kz, zposts; σz = σ)

    # update reservoir
    step!(hsic.reservoir,
          state,
          hsic.learner,
          concatenate(x, y, zpost),
          ξ, t, Δt; explore = explore)

    # compute local terms
    @cast β[i, j] := (1 - zpost[i]^2) * zpre[j]

    # compute weight update
    @cast Δw[i, j] := -state.z[i] * β[i, j]

    return Δw
end

function update!(η, layer::LIFDense, hsic::HSIC, state, x, y, zpre, zpost, t, Δt)
    layer.W .+= η(t) .* hsic(state, x, y, zpre, zpost, t, Δt)
end

function update!(η, layers::LIFChain, hsics, states, x, y, zpres, zposts, t, Δt)
    foreach(layers, hsics, states, zpres, zposts) do layer, hsic, state, zpre, zpost
        update!(η, layer, hsic, state, x, y, zpre, zpost, t, Δt)
    end
end
