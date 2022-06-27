struct HSICApprox{T<:KernelMatrices, S<:GlobalError}
    kernels::T
    error::S
end

function HSICApprox(::Type{T}, Din, Dout, Dlayer, bs::Integer;
                    γ, nbuffer, sigmas) where T
    kernels = KernelMatrices(T, map(T, sigmas), (Din, Dout, Dlayer), bs, nbuffer)
    error = GlobalError{T}(bs, prod(Dlayer); λ = γ)

    return HSICApprox(kernels, error)
end

Adapt.adapt_structure(to, hsic::HSICApprox) =
    HSICApprox(adapt(to, hsic.kernels), adapt(to, hsic.error))

cpu(hsic::HSICApprox) = adapt(Array, hsic)
gpu(hsic::HSICApprox) = adapt(CuArray, hsic)

state(::HSICApprox) = nothing

function (hsic::HSICApprox)(x, y, zpost)
    # update kernel matrices
    Kx, Ky, Kz = push!(hsic.kernels, x, y, zpost)
    zposts = hsic.kernels.caches[3]

    # compute global term
    return hsic.error(Kx, Ky, Kz, zposts; σz = hsic.kernels.sigmas[3])
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
