struct Reservoir{T, S<:AbstractMatrix{T}, F, G, H}
    Wr::S
    Win::S
    Wfb::S
    Wout::S
    ξhidden::F
    ξ::G
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
    Dξhidden = Uniform(-0.05, 0.05)
    Dξ = Uniform(-0.5, 0.5)

    # initial values
    Wr = convert.(T, rand(Dp, nhidden, nhidden) .* rand(Dr, nhidden, nhidden))
    Win = convert.(T, rand(Din, nhidden, nin))
    Wfb = convert.(T, rand(Dfb, nhidden, nout))
    Wout = zeros(T, nout, nhidden)

    # activation function
    f = tanh

    S = typeof(Wr)
    F = typeof(Dξhidden)
    G = typeof(Dξ)
    H = typeof(f)

    Reservoir{T, S, F, G, H}(Wr, Win, Wfb, Wout, Dξhidden, Dξ, f, τ, λ)
end
Reservoir{T}(inout::Pair, nhidden; kwargs...) where T =
    Reservoir{T}(inout[1], inout[2], nhidden; kwargs...)

nin(reservoir::Reservoir) = size(reservoir.Win, 2)
nout(reservoir::Reservoir) = size(reservoir.Wout, 1)
nhidden(reservoir::Reservoir) = size(reservoir.Wfb, 1)

state(reservoir::Reservoir{T}) where T = zeros(T, nhidden(reservoir))

struct ReservoirCache{T}
    r::T
    z::T
end
ReservoirCache(reservoir::Reservoir{T}) where T =
    ReservoirCache(zeros(T, nhidden(reservoir)), zeros(T, nout(reservoir)))

function (reservoir::Reservoir)(du, u, p, t)
    input, cache = p

    cache.r .= reservoir.f.(u) + rand(reservoir.ξhidden, nhidden(reservoir))
    cache.z .= reservoir.Wout * cache.r + rand(reservoir.ξ, nout(reservoir))
    du .= (-u + reservoir.λ .* reservoir.Wr * cache.r +
                reservoir.Win * input(t) +
                reservoir.Wfb * cache.z) / reservoir.τ

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
RFORCE(reservoir::Reservoir{T}; η, τ) where T = RFORCE{T}(η, τ, nout(reservoir))

function (learner::RFORCE)(reservoir::Reservoir, r, z, f, Δt)
    P = -norm(z .- f)^2
    P̄ = learner.Plpf(P, Δt)
    M = P > only(P̄)

    z̄ = learner.zlpf(z, Δt)

    reservoir.Wout .+= learner.η * transpose(z .- z̄) * M * r

    return reservoir
end
(learner::RFORCE)(integrator::DifferentialEquations.DiffEqBase.DEIntegrator, f, Δt) =
    learner(integrator.f, integrator.p.cache.r, integrator.p.cache.z, f, Δt)
    