struct GlobalError{T, S, R, Q}
    γ::T
    α::S
    ξ::R
    λ::Q
end
GlobalError{T}(bs, n; λ = 2) where {T} =
    GlobalError(zeros(T, bs), zeros(T, bs, n), zeros(T, n), λ)

Adapt.adapt_structure(to, error::GlobalError) = GlobalError(adapt(to, error.γ),
                                                            adapt(to, error.α),
                                                            adapt(to, error.ξ),
                                                            error.λ)
cpu(error::GlobalError) = adapt(Array, error)
gpu(error::GlobalError) = adapt(CuArray, error)

function (error::GlobalError)(kx, ky, kz, z; σz = estσ(z))
    bs = size(error.γ, 1)

    @cast error.γ[i] = (kx[1, i] - @reduce sum(k) kx[1, k] / bs) -
                        error.λ * (ky[1, i] - @reduce sum(k) ky[1, k] / bs)

    @cast error.α[i, k] = -2 * kz[1, i] * (z[k, 1] - z[k, i]) / σz^2
    @cast error.α[i, k] = error.α[i, k] - @reduce _[k] := sum(n) error.α[n, k] / bs

    @reduce error.ξ[k] = sum(i) error.γ[i] * error.α[i, k] / (bs - 1)^2

    return error.ξ
end

function _fill!(buffer::CircularBuffer{T}, xs::AbstractVector{T}) where T
    foreach(x -> isfull(buffer) || push!(buffer, x), xs)

    return buffer
end

CircularBuffer(xs::AbstractVector{T}) where T =
    _fill!(CircularBuffer{T}(length(xs)), xs)

function _shiftbuffers!(buffers, sample)
    @inbounds next = popfirst!.(buffers[1:(end - 1)])
    @inbounds push!(buffers[1], sample)
    @inbounds push!.(buffers[2:end], next)

    return buffers
end

struct HSICIdeal{T, S, R, P}
    Xs::T
    Ys::T
    Zpres::T
    Zposts::T
    Kx::S
    Ky::S
    Kz::S
    γ::R
    cache::P
end

function HSICIdeal(xdata, ydata, layer::LIFDense{T}, ::LIFDenseState{S}, bs; γ, nbuffer) where {T, S}
    n = nout(layer)
    Din = size(xdata, 1)
    Dout = size(ydata, 1)
    XT = eltype(xdata)
    YT = eltype(ydata)
    ZT = eltype(S)

    Xs = [fill!(CircularBuffer{Vector{XT}}(nbuffer), zeros(XT, Din)) for _ in 1:bs]
    Ys = [fill!(CircularBuffer{Vector{YT}}(nbuffer), zeros(YT, Dout)) for _ in 1:bs]
    Zpres = [fill!(CircularBuffer{Vector{ZT}}(nbuffer), zeros(ZT, nin(layer))) for _ in 1:bs]
    Zposts = [fill!(CircularBuffer{Vector{ZT}}(nbuffer), zeros(ZT, n)) for _ in 1:bs]
    cache = (x = zeros(XT, Din, bs),
             y = zeros(YT, Dout, bs),
             zpre = zeros(ZT, nin(layer), bs),
             zpost = zeros(ZT, n, bs))

    Kx = zeros(XT, bs, bs)
    Ky = zeros(YT, bs, bs)
    Kz = zeros(ZT, bs, bs)

    return HSICIdeal(Xs, Ys, Zpres, Zposts, Kx, Ky, Kz, γ, cache)
end

function Adapt.adapt_structure(to, hsic::HSICIdeal)
    Xs = [CircularBuffer(adapt.(to, x)) for x in hsic.Xs]
    Ys = [CircularBuffer(adapt.(to, x)) for x in hsic.Ys]
    Zpres = [CircularBuffer(adapt.(to, x)) for x in hsic.Zpres]
    Zposts = [CircularBuffer(adapt.(to, x)) for x in hsic.Zposts]
    cache = (x = adapt(to, hsic.cache.x),
             y = adapt(to, hsic.cache.y),
             zpre = adapt(to, hsic.cache.zpre),
             zpost = adapt(to, hsic.cache.zpost))
    Kx = adapt(to, hsic.Kx)
    Ky = adapt(to, hsic.Ky)
    Kz = adapt(to, hsic.Kz)

    return HSICIdeal(Xs, Ys, Zpres, Zposts, Kx, Ky, Kz, hsic.γ, cache)
end

cpu(hsic::HSICIdeal) = adapt(Array, hsic)
gpu(hsic::HSICIdeal) = adapt(CuArray, hsic)

state(::HSICIdeal) = nothing

function (hsic::HSICIdeal)(x, y, zpre, zpost)
    xs, ys, zpres, zposts = hsic.cache

    # push new samples into buffers
    _shiftbuffers!(hsic.Xs, x)
    _shiftbuffers!(hsic.Ys, y)
    _shiftbuffers!(hsic.Zpres, zpre)
    _shiftbuffers!(hsic.Zposts, zpost)

    # compute new kernel matrices
    for n in 1:size(xs, 2)
        @inbounds xs[:, n] .= hsic.Xs[n][1]
        @inbounds ys[:, n] .= hsic.Ys[n][1]
        @inbounds zpres[:, n] .= hsic.Zpres[n][1]
        @inbounds zposts[:, n] .= hsic.Zposts[n][1]
    end
    σ = max(1f-3, estσ(zposts))
    k_hsic!(hsic.Kx, xs, xs; σ = max(1f-3, estσ(xs)))
    k_hsic!(hsic.Ky, ys, ys; σ = (size(ys, 1) == 1) ? 5f-1 : 1f0)
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
(hsic::HSICIdeal)(state, x, y, zpre, zpost, t, Δt) = hsic(x, y, zpre, zpost)

struct HSICApprox{T, S, R, E<:GlobalError, U}
    Xs::T
    Ys::T
    Zposts::T
    Kx::S
    Ky::S
    Kz::S
    γ::R
    ξ::E
    cache::U
end

function HSICApprox(xdata, ydata, layer::LIFDense{T}, ::LIFDenseState{S}, bs; γ, nbuffer) where {T, S}
    n = nout(layer)
    Din = size(xdata, 1)
    Dout = size(ydata, 1)
    XT = eltype(xdata)
    YT = eltype(ydata)
    ZT = eltype(S)

    Xs = [fill!(CircularBuffer{Vector{XT}}(nbuffer), zeros(XT, Din)) for _ in 1:bs]
    Ys = [fill!(CircularBuffer{Vector{YT}}(nbuffer), zeros(YT, Dout)) for _ in 1:bs]
    Zposts = [fill!(CircularBuffer{Vector{ZT}}(nbuffer), zeros(ZT, n)) for _ in 1:bs]
    cache = (x = zeros(XT, Din, bs),
             y = zeros(YT, Dout, bs),
             zpost = zeros(ZT, n, bs))

    Kx = zeros(XT, bs, bs)
    Ky = zeros(YT, bs, bs)
    Kz = zeros(ZT, bs, bs)

    ξ = GlobalError{T}(bs, n; λ = γ)

    return HSICApprox(Xs, Ys, Zposts, Kx, Ky, Kz, γ, ξ, cache)
end

function Adapt.adapt_structure(to, hsic::HSICApprox)
    Xs = [CircularBuffer(adapt.(to, x)) for x in hsic.Xs]
    Ys = [CircularBuffer(adapt.(to, x)) for x in hsic.Ys]
    Zposts = [CircularBuffer(adapt.(to, x)) for x in hsic.Zposts]
    cache = (x = adapt(to, hsic.cache.x),
             y = adapt(to, hsic.cache.y),
             zpost = adapt(to, hsic.cache.zpost))
    Kx = adapt(to, hsic.Kx)
    Ky = adapt(to, hsic.Ky)
    Kz = adapt(to, hsic.Kz)
    ξ = adapt(to, hsic.ξ)

    return HSICApprox(Xs, Ys, Zposts, Kx, Ky, Kz, hsic.γ, ξ, cache)
end

cpu(hsic::HSICApprox) = adapt(Array, hsic)
gpu(hsic::HSICApprox) = adapt(CuArray, hsic)

state(::HSICApprox) = nothing

function (hsic::HSICApprox)(x, y, zpre, zpost)
    xs, ys, zposts = hsic.cache

    # push new samples into buffers
    _shiftbuffers!(hsic.Xs, x)
    _shiftbuffers!(hsic.Ys, y)
    _shiftbuffers!(hsic.Zposts, zpost)

    # compute new kernel matrices
    for n in 1:size(xs, 2)
        @inbounds xs[:, n] .= hsic.Xs[n][1]
        @inbounds ys[:, n] .= hsic.Ys[n][1]
        @inbounds zposts[:, n] .= hsic.Zposts[n][1]
    end
    σ = max(1f-3, estσ(zposts))
    k_hsic!(hsic.Kx, xs, xs; σ = max(1f-3, estσ(xs)))
    k_hsic!(hsic.Ky, ys, ys; σ = (size(ys, 1) == 1) ? 5f-1 : 1f0)
    k_hsic!(hsic.Kz, zposts, zposts; σ = σ)

    # compute global term
    ξ = hsic.ξ(hsic.Kx, hsic.Ky, hsic.Kz, zposts; σz = σ)

    # compute local terms
    @cast β[i, j] := (1 - zpost[i]^2) * zpre[j]

    # compute weight update
    @cast Δw[i, j] := -ξ[i] * β[i, j]

    return Δw
end
(hsic::HSICApprox)(state, x, y, zpre, zpost, t, Δt) = hsic(x, y, zpre, zpost)

struct HSIC{T, S, R, E<:GlobalError, P<:Reservoir, Q<:RMHebb, U}
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
    cache::U
end

function HSIC(xdata,
              ydata,
              layer::LIFDense{T},
              ::LIFDenseState{S},
              reservoir::Reservoir,
              learner::RMHebb,
              bs; γ, nbuffer) where {T, S}
    n = nout(layer)
    Din = size(xdata, 1)
    Dout = size(ydata, 1)
    XT = eltype(xdata)
    YT = eltype(ydata)
    ZT = eltype(S)

    Xs = [fill!(CircularBuffer{Vector{XT}}(nbuffer), zeros(XT, Din)) for _ in 1:bs]
    Ys = [fill!(CircularBuffer{Vector{YT}}(nbuffer), zeros(YT, Dout)) for _ in 1:bs]
    Zposts = [fill!(CircularBuffer{Vector{ZT}}(nbuffer), zeros(ZT, n)) for _ in 1:bs]
    cache = (x = zeros(XT, Din, bs),
             y = zeros(YT, Dout, bs),
             zpost = zeros(ZT, n, bs))

    Kx = zeros(XT, bs, bs)
    Ky = zeros(YT, bs, bs)
    Kz = zeros(ZT, bs, bs)

    ξ = GlobalError{T}(bs, n; λ = γ)

    return HSIC(Xs, Ys, Zposts, Kx, Ky, Kz, γ, ξ, reservoir, learner, cache)
end

function Adapt.adapt_structure(to, hsic::HSIC)
    Xs = [CircularBuffer(adapt.(to, x)) for x in hsic.Xs]
    Ys = [CircularBuffer(adapt.(to, x)) for x in hsic.Ys]
    Zposts = [CircularBuffer(adapt.(to, x)) for x in hsic.Zposts]
    cache = (x = adapt(to, hsic.cache.x),
             y = adapt(to, hsic.cache.y),
             zpost = adapt(to, hsic.cache.zpost))
    Kx = adapt(to, hsic.Kx)
    Ky = adapt(to, hsic.Ky)
    Kz = adapt(to, hsic.Kz)
    ξ = adapt(to, hsic.ξ)
    reservoir = adapt(to, hsic.reservoir)
    learner = adapt(to, hsic.learner)

    return HSIC(Xs, Ys, Zposts, Kx, Ky, Kz, hsic.γ, ξ, reservoir, learner, cache)
end

cpu(hsic::HSIC) = adapt(Array, hsic)
gpu(hsic::HSIC) = adapt(CuArray, hsic)

state(hsic::HSIC) = state(hsic.reservoir)

function (hsic::HSIC)(state, x, y, zpre, zpost, t, Δt)
    xs, ys, zposts = hsic.cache

    # push new samples into buffers
    _shiftbuffers!(hsic.Xs, x)
    _shiftbuffers!(hsic.Ys, y)
    _shiftbuffers!(hsic.Zposts, zpost)

    # compute new kernel matrices
    xs .= reduce(hcat, getindex.(hsic.Xs, 1))
    ys .= reduce(hcat, getindex.(hsic.Ys, 1))
    zposts .= reduce(hcat, getindex.(hsic.Zposts, 1))
    σ = max(1f-3, estσ(zposts))
    k_hsic!(hsic.Kx, xs, xs; σ = max(1f-3, estσ(xs)))
    k_hsic!(hsic.Ky, ys, ys; σ = (size(ys, 1) == 1) ? 5f-1 : 1f0)
    k_hsic!(hsic.Kz, zposts, zposts; σ = σ)

    # compute global term
    ξ = hsic.ξ(hsic.Kx, hsic.Ky, hsic.Kz, zposts; σz = σ)

    # update reservoir
    step!(hsic.reservoir,
          state,
          hsic.learner,
          concatenate(x, y, zpost),
          ξ, t, Δt)

    # compute local terms
    @cast β[i, j] := (1 - zpost[i]^2) * zpre[j]

    # compute weight update
    @cast Δw[i, j] := -state.z[i] * β[i, j]

    return Δw
end

function update!(η, layer::LIFDense, hsic, state, x, y, zpre, zpost, t, Δt)
    layer.W .+= η(t) .* hsic(state, x, y, zpre, zpost, t, Δt)
end

function update!(η, layers::LIFChain, hsics, states, x, y, zpres, zposts, t, Δt; nthreads = 1)
    if nthreads > 1
        Threads.@threads for i in 1:length(layers)
            update!(η, layers[i], hsics[i], states[i], x, y, zpres[i], zposts[i], t, Δt)
        end
    else
        foreach(layers, hsics, states, zpres, zposts) do layer, hsic, state, zpre, zpost
            update!(η, layer, hsic, state, x, y, zpre, zpost, t, Δt)
        end
    end
end
