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

function (error::GlobalError)(kx, ky, kz, z; σz = estσ(z))
    bs = size(error.γ, 1)

    @cast error.γ[i] = (kx[i] - @reduce sum(k) kx[k] / bs) -
                        error.λ * (ky[i] - @reduce sum(k) ky[k] / bs)


    @cast error.α[i, k] = -2 * kz[i] * (z[k, 1] - z[k, i]) / σz^2
    @cast error.α[i, k] = error.α[i, k] - @reduce _[k] := sum(n) error.α[n, k] / bs

    @reduce error.ξ[k] = sum(i) error.γ[i] * error.α[i, k] / (bs - 1)^2

    return error.ξ
end
function (error::GlobalError)(Kx::AbstractMatrix, Ky::AbstractMatrix, Kz::AbstractMatrix, z; kwargs...)
    kx = view(Kx, 1, :)
    ky = view(Ky, 1, :)
    kz = view(Kz, 1, :)

    return error(kx, ky, kz, z; kwargs...)
end

struct HSICApprox{T<:NTuple{<:Any, <:KernelCache}, S<:GlobalError}
    kernels::T
    error::S
end

function HSICApprox(::Type{T}, Din, Dout, Dlayer, bs::Integer;
                    γ, nbuffer, sigmas) where T
    kernels = (KernelCache(T, Din, bs, nbuffer, sigmas[1]),
               KernelCache(T, Dout, bs, nbuffer, sigmas[2]),
               KernelCache(T, Dlayer, bs, nbuffer, sigmas[3]))
    error = GlobalError{T}(bs, prod(Dlayer); λ = γ)

    return HSICApprox(kernels, error)
end

Adapt.@adapt_structure HSICApprox

state(::HSICApprox) = nothing

function (hsic::HSICApprox)(x, y, zpost)
    # update kernel matrices
    Kx = push!(hsic.kernels[1], x)
    Ky = push!(hsic.kernels[2], y)
    Kz = push!(hsic.kernels[3], zpost)
    zposts = activity_cache(hsic.kernels[3])

    # compute global term
    return hsic.error(Kx, Ky, Kz, zposts; σz = hsic.kernels[3].sigma)
end
(hsic::HSICApprox)(state, x, y, zpre, zpost, t, Δt) = hsic(x, y, zpost)

struct HSICReservoir{T<:KernelMatrices, S<:GlobalError, R<:Reservoir, Q<:RMHebb}
    kernels::T
    error::S
    reservoir::R
    learner::Q
end

function HSICReservoir(::Type{T}, Din::Integer, Dout::Integer, Dlayer::Integer, bs::Integer;
                       η, τr, τlpf, λ, noise, Dhidden, γ, sigmas, nbuffer) where T
    kernels = KernelMatrices(T, sigmas, (Din, Dout, Dlayer), bs, nbuffer)
    error = GlobalError{T}(bs, Dlayer; λ = γ)
    reservoir = Reservoir{T}(Din + Dlayer + Dout => Dlayer, Dhidden;
                             τ = τr, λ = λ, noiseout = noise)
    learner = RMHebb(reservoir; η = η, τ = τlpf)

    return HSICReservoir(kernels, error, reservoir, learner)
end

Adapt.adapt_structure(to, hsic::HSICReservoir) =
    HSICReservoir(adapt(to, hsic.kernels),
                  adapt(to, hsic.errror),
                  adapt(to, hsic.reservoir),
                  adapt(to, hsic.learner))

cpu(hsic::HSICReservoir) = adapt(Array, hsic)
gpu(hsic::HSICReservoir) = adapt(CuArray, hsic)

state(hsic::HSICReservoir) = state(hsic.reservoir)

function (hsic::HSICReservoir)(state, x, y, zpost, t, Δt)
    # update kernel matrices
    Kx, Ky, Kz = push!(hsic.kernels, x, y, zpost)
    zposts = hsic.kernels.caches[3]

    # compute global term
    ξ = 1000 * hsic.error(Kx, Ky, Kz, zposts; σz = hsic.kernels.sigmas[3])

    # update reservoir
    step!(hsic.reservoir, state, hsic.learner, concatenate(x, y, zpost), ξ, t, Δt)

    # return reservoir output
    return -state.z[i]
end
(hsic::HSICReservoir)(state, x, y, zpre, zpost, t, Δt) = hsic(state, x, y, zpost, t, Δt)
