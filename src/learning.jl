

# function _fill!(buffer::CircularBuffer{T}, xs::AbstractVector{T}) where T
#     foreach(x -> isfull(buffer) || push!(buffer, x), xs)

#     return buffer
# end

# CircularBuffer(xs::AbstractVector{T}) where T =
#     _fill!(CircularBuffer{T}(length(xs)), xs)

# _adapt(::Type{<:Array}, xs::Vector{<:CircularBuffer}) = map(x -> CircularBuffer(Array.(x)), xs)
# _adapt(::Type{<:CuArray}, xs::Vector{<:CircularBuffer}) = map(x -> CircularBuffer(cu.(x)), xs)

# function _shiftbuffers!(buffers, sample)
#     @inbounds next = popfirst!.(buffers[1:(end - 1)])
#     @inbounds push!(buffers[1], sample)
#     @inbounds push!.(buffers[2:end], next)

#     return buffers
# end

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
    σ = estσ(zposts)
    k_hsic!(hsic.Kx, xs, xs; σ = estσ(xs))
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

function HSICApprox{T}(Din::Integer, Dout::Integer, Dlayer::Integer, bs::Integer;
                       γ, nbuffer) where T
    Xs = preload!(TimeBatch(T, Din; batchsize = bs, sample_length = nbuffer),
                  zeros(T, Din))
    Ys = preload!(TimeBatch(T, Dout; batchsize = bs, sample_length = nbuffer),
                  zeros(T, Dout))
    Zposts = preload!(TimeBatch(T, Dlayer; batchsize = bs, sample_length = nbuffer),
                      zeros(T, Dlayer))
    cache = (x = zeros(T, Din, bs),
             y = zeros(T, Dout, bs),
             zpost = zeros(T, Dlayer, bs))

    Kx = zeros(T, bs, bs)
    Ky = zeros(T, bs, bs)
    Kz = zeros(T, bs, bs)

    ξ = GlobalError{T}(bs, Dlayer; λ = γ)

    return HSICApprox(Xs, Ys, Zposts, Kx, Ky, Kz, γ, ξ, cache)
end

function Adapt.adapt_structure(to, hsic::HSICApprox)
    Xs = adapt(to, hsic.Xs)
    Ys = adapt(to, hsic.Ys)
    Zposts = adapt(to, hsic.Zposts)
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
    push!(hsic.Xs, x)
    push!(hsic.Ys, y)
    push!(hsic.Zposts, zpost)

    # compute new kernel matrices
    n = size(xs, 2)
    @inbounds xs .= view(hsic.Xs, 0:-1:-(n - 1), 0)
    @inbounds ys .= view(hsic.Ys, 0:-1:-(n - 1), 0)
    @inbounds zposts .= view(hsic.Zposts, 0:-1:-(n - 1), 0)
    σx = 2f-1 * sqrt(size(xs, 1))
    σy = (size(ys, 1) == 1) ? 5f-1 : 1f0
    σz = 5f-1 * sqrt(size(zposts, 1))
    k_hsic!(hsic.Kx, xs; σ = σx)
    k_hsic!(hsic.Ky, ys; σ = σy)
    k_hsic!(hsic.Kz, zposts; σ = σz)

    # compute global term
    ξ = hsic.ξ(hsic.Kx, hsic.Ky, hsic.Kz, zposts; σz = σz)

    # compute weight update
    @cast Δw[i, j] := ξ[i] * (1 - zpost[i]^2) * zpre[j]

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
    σ = estσ(zposts)
    k_hsic!(hsic.Kx, xs, xs; σ = estσ(xs))
    k_hsic!(hsic.Ky, ys, ys; σ = (size(ys, 1) == 1) ? 5f-1 : 1f0)
    k_hsic!(hsic.Kz, zposts, zposts; σ = σ)

    # compute global term
    ξ = 1000 * hsic.ξ(hsic.Kx, hsic.Ky, hsic.Kz, zposts; σz = σ)

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

struct pHSIC{T, S, R, U}
    Ys::T
    Zpres::T
    Zposts::T
    Ky::S
    Kz::S
    γ::R
    cache::U
end

function pHSIC(ydata, layer::LIFDense{T}, ::LIFDenseState{S}; γ, nbuffer) where {T, S}
    bs = 2
    n = nout(layer)
    Dout = size(ydata, 1)
    YT = eltype(ydata)
    ZT = eltype(S)

    Ys = [fill!(CircularBuffer{Vector{YT}}(nbuffer), zeros(YT, Dout)) for _ in 1:bs]
    Zpres = [fill!(CircularBuffer{Vector{ZT}}(nbuffer), zeros(ZT, nin(layer))) for _ in 1:bs]
    Zposts = [fill!(CircularBuffer{Vector{ZT}}(nbuffer), zeros(ZT, n)) for _ in 1:bs]
    cache = (y = zeros(YT, Dout, bs),
             zpre = zeros(ZT, nin(layer), bs),
             zpost = zeros(ZT, n, bs))

    Ky = zeros(YT, bs, bs)
    Kz = zeros(ZT, bs, bs)

    return pHSIC(Ys, Zpres, Zposts, Ky, Kz, γ, cache)
end

function Adapt.adapt_structure(to, hsic::pHSIC)
    Ys = [CircularBuffer(adapt.(to, x)) for x in hsic.Ys]
    Zpres = [CircularBuffer(adapt.(to, x)) for x in hsic.Zpres]
    Zposts = [CircularBuffer(adapt.(to, x)) for x in hsic.Zposts]
    cache = (y = adapt(to, hsic.cache.y),
             zpre = adapt(to, hsic.cache.zpre),
             zpost = adapt(to, hsic.cache.zpost))
    Ky = adapt(to, hsic.Ky)
    Kz = adapt(to, hsic.Kz)

    return pHSIC(Ys, Zpres, Zposts, Ky, Kz, hsic.γ, cache)
end

cpu(hsic::pHSIC) = adapt(Array, hsic)
gpu(hsic::pHSIC) = adapt(CuArray, hsic)

state(::pHSIC) = nothing

function (hsic::pHSIC)(y, zpre, zpost)
    ys, zpres, zposts = hsic.cache

    # push new samples into buffers
    _shiftbuffers!(hsic.Ys, y)
    _shiftbuffers!(hsic.Zpres, zpre)
    _shiftbuffers!(hsic.Zposts, zpost)

    # compute new kernel matrices
    for n in 1:size(ys, 2)
        @inbounds ys[:, n] .= hsic.Ys[n][1]
        @inbounds zpres[:, n] .= hsic.Zpres[n][1]
        @inbounds zposts[:, n] .= hsic.Zposts[n][1]
    end
    σy = (size(ys, 1) == 1) ? 5f-1 : 1f0
    σz = 7.5f-1 * sqrt(size(zposts, 1))
    # σz = 5f0
    k_hsic!(hsic.Ky, ys; σ = σy)
    k_hsic!(hsic.Kz, zposts; σ = σz)

    K̄z = hsic.Kz .- mean(hsic.Kz; dims = 2)
    K̄y = hsic.Ky .- mean(hsic.Ky; dims = 2)
    M = mean((hsic.γ .* K̄y .- 2 .* K̄z) .* hsic.Kz ./ σz^2)

    # compute weight update
    Δw = M * (zposts[:, 1] .- zposts[:, 2]) * transpose(zpres[:, 1] .- zpres[:, 2])
    # @reduce Δw[i, j] = sum(p, q) (  2      * (hsic.Kz[p, q] - @reduce _[p] := sum(n) hsic.Kz[p, n] / 2)
    #                                - hsic.γ * (hsic.Ky[p, q] - @reduce _[p] := sum(n) hsic.Ky[p, n] / 2)) *
    #                                -(hsic.Kz[p, q] / σz^2) * 
    #                                 (zposts[i, p] - zposts[i, q]) *
    #                                 ((1 - zposts[i, p])^2 * zpres[j, p] - (1 - zposts[i, q])^2 * zpres[j, q])

    return Δw
end
(hsic::pHSIC)(state, x, y, zpre, zpost, t, Δt) = hsic(y, zpre, zpost)
