struct Reservoir{T, S<:AbstractMatrix{T}, F, G, H}
    Wr::S
    Win::S
    Wfb::S
    Wout::S
    ξhidden!::F
    ξ!::G
    f::H
    τ::T
    λ::T
end
function Reservoir{T}(nin, nout, nhidden; τ = 10e-3, λ = 1.2, sparsity = 0.1) where T
    # network parameters
    Dp = Bernoulli(sparsity) # probability of recurrent connection
    Dr = Normal(0, 1 / (sparsity * nhidden)) # weight distribution of recurrent connection
    Din = Uniform(-1, 1) # weight distribution of input connection
    Dfb = Uniform(-1, 1) # weight distribution of feedback connection

    # noise distributions
    function ξhidden!(dest) # Uniform(-0.05, 0.05)
        rand!(dest)
        @. dest = 0.1 * dest - 0.05
        return dest
    end
    function ξ!(dest) # Uniform(-0.5, 0.5)
        rand!(dest)
        @. dest = dest - 0.5
        return dest
    end

    # initial values
    Wr = convert.(T, rand(Dp, nhidden, nhidden) .* rand(Dr, nhidden, nhidden))
    Win = convert.(T, rand(Din, nhidden, nin))
    Wfb = convert.(T, rand(Dfb, nhidden, nout))
    Wout = zeros(T, nout, nhidden)

    # activation function
    f = tanh

    S = typeof(Wr)
    F = typeof(ξhidden!)
    G = typeof(ξ!)
    H = typeof(f)

    Reservoir{T, S, F, G, H}(Wr, Win, Wfb, Wout, ξhidden!, ξ!, f, τ, λ)
end
Reservoir{T}(inout::Pair, nhidden; kwargs...) where T =
    Reservoir{T}(inout[1], inout[2], nhidden; kwargs...)

nin(reservoir::Reservoir) = size(reservoir.Win, 2)
nout(reservoir::Reservoir) = size(reservoir.Wout, 1)
nhidden(reservoir::Reservoir) = size(reservoir.Wfb, 1)

state(reservoir::Reservoir{T, S}) where {T, S} = adapt(S, zeros(nhidden(reservoir)))

cpu(reservoir::Reservoir) = Reservoir(adapt(Array, reservoir.Wr),
                                      adapt(Array, reservoir.Win),
                                      adapt(Array, reservoir.Wfb),
                                      adapt(Array, reservoir.Wout),
                                      reservoir.ξhidden!,
                                      reservoir.ξ!,
                                      reservoir.f,
                                      reservoir.τ,
                                      reservoir.λ)
gpu(reservoir::Reservoir) = Reservoir(adapt(CuArray, reservoir.Wr),
                                      adapt(CuArray, reservoir.Win),
                                      adapt(CuArray, reservoir.Wfb),
                                      adapt(CuArray, reservoir.Wout),
                                      reservoir.ξhidden!,
                                      reservoir.ξ!,
                                      reservoir.f,
                                      reservoir.τ,
                                      reservoir.λ)

struct ReservoirCache{T}
    r::T
    z::T
end
ReservoirCache(reservoir::Reservoir{T}) where T =
    ReservoirCache(similar(reservoir.Wr, nhidden(reservoir)),
                   similar(reservoir.Wr, nout(reservoir)))

# CPU implementation
function (reservoir::Reservoir{T, S})(du, u, p, t) where {T, S<:Array}
    input, cache = p

    # get hidden neuron firing rate
    reservoir.ξhidden!(cache.r)
    @avx cache.r .+= reservoir.f.(u)
    
    # get output neuron firing rate
    reservoir.ξ!(cache.z)
    @avx cache.z .+= reservoir.Wout * cache.r
    
    # update du
    @avx du .= (-u .+ reservoir.λ .* reservoir.Wr * cache.r .+
                      reservoir.Win * input(t) .+
                      reservoir.Wfb * cache.z) ./ reservoir.τ

    return du
end
# GPU implementation
function (reservoir::Reservoir{T})(du, u, p, t) where T
    input, cache = p

    # get hidden neuron firing rate
    reservoir.ξhidden!(cache.r)
    cache.r .+= reservoir.f.(u)
    
    # get output neuron firing rate
    reservoir.ξ!(cache.z)
    cache.z .+= reservoir.Wout * cache.r
    
    # update du
    du .= (-u .+ reservoir.λ .* reservoir.Wr * cache.r .+
                 reservoir.Win * input(t) .+
                 reservoir.Wfb * cache.z) ./ reservoir.τ

    return du
end

struct RFORCE{T, S, R}
    η::T
    zlpf::S
    Plpf::R
end
function RFORCE{T}(η, τ, N) where T
    zlpf = LowPassFilter{T, Vector{T}}(τ, zeros(T, N))
    Plpf = LowPassFilter{T, Vector{T}}(τ, zeros(T, 1))

    RFORCE{T, typeof(zlpf), typeof(Plpf)}(η, zlpf, Plpf)
end
RFORCE(reservoir::Reservoir{T}; η, τ) where {T} = RFORCE{T}(η, τ, nout(reservoir))

cpu(learner::RFORCE) = RFORCE(learner.η,
                              LowPassFilter(learner.zlpf.τ, adapt(Array, learner.zlpf.f̄)),
                              LowPassFilter(learner.Plpf.τ, adapt(Array, learner.Plpf.f̄)))
gpu(learner::RFORCE) = RFORCE(learner.η,
                              LowPassFilter(learner.zlpf.τ, adapt(CuArray, learner.zlpf.f̄)),
                              LowPassFilter(learner.Plpf.τ, adapt(CuArray, learner.Plpf.f̄)))

# CPU implementation
function (learner::RFORCE)(reservoir::Reservoir{T, S}, r, z, f, Δt) where {T, S<:Array}
    P = -norm(z .- f)^2
    P̄ = learner.Plpf(P, Δt)
    M = adapt(typeof(z), @. P > P̄)

    z̄ = learner.zlpf(z, Δt)

    @avx reservoir.Wout .+= learner.η * (z .- z̄) .* M * transpose(r)

    return reservoir
end
# GPU implementation
function (learner::RFORCE)(reservoir::Reservoir, r, z, f, Δt)
    P = -norm(z .- f)^2
    P̄ = learner.Plpf(P, Δt)
    M = adapt(typeof(z), @. P > P̄)

    z̄ = learner.zlpf(z, Δt)

    reservoir.Wout .+= learner.η * (z .- z̄) .* M * transpose(r)

    return reservoir
end
(learner::RFORCE)(integrator::DifferentialEquations.DiffEqBase.DEIntegrator, f, Δt) =
    learner(integrator.f.f, integrator.p.cache.r, integrator.p.cache.z, f(integrator.t), Δt)
    