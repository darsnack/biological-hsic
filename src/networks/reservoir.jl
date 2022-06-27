struct Reservoir{T, S<:AbstractMatrix{T}, F, G}
    Wr::S
    Win::S
    Wfb::S
    Wout::S
    ξhidden!::F
    ξ!::G
    τ::T
    λ::T
end
function Reservoir{T}(nin, nout, nhidden;
                      τ = 10e-3, λ = 1.2,
                      sparsity = 0.1, noisehidden = 0.05, noiseout = 0.5) where T
    # network parameters
    Dp = Bernoulli(sparsity) # probability of recurrent connection
    Dr = Normal(0, sqrt(1 / (sparsity * nhidden))) # weight distribution of recurrent connection
    Din = Uniform(-1, 1) # weight distribution of input connection
    Dfb = Uniform(-1, 1) # weight distribution of feedback connection

    # noise distributions
    function ξhidden!(dest) # Uniform(-0.05, 0.05)
        rand!(dest)
        @. dest = 2 * noisehidden * dest - noisehidden
        return dest
    end
    function ξ!(dest) # Uniform(-0.5, 0.5)
        rand!(dest)
        @. dest = 2 * noiseout * dest - noiseout
        return dest
    end

    # initial values
    Wr = convert.(T, rand(Dp, nhidden, nhidden) .* rand(Dr, nhidden, nhidden))
    Win = convert.(T, rand(Din, nhidden, nin))
    Wfb = convert.(T, rand(Dfb, nhidden, nout))
    Wout = zeros(T, nout, nhidden)

    S = typeof(Wr)
    F = typeof(ξhidden!)
    G = typeof(ξ!)

    Reservoir{T, S, F, G}(Wr, Win, Wfb, Wout, ξhidden!, ξ!, τ, λ)
end
Reservoir{T}(inout::Pair, nhidden; kwargs...) where {T} =
    Reservoir{T}(inout[1], inout[2], nhidden; kwargs...)

Adapt.adapt_structure(to, reservoir::Reservoir) = Reservoir(adapt(to, reservoir.Wr),
                                                            adapt(to, reservoir.Win),
                                                            adapt(to, reservoir.Wfb),
                                                            adapt(to, reservoir.Wout),
                                                            reservoir.ξhidden!,
                                                            reservoir.ξ!,
                                                            reservoir.τ,
                                                            reservoir.λ)

cpu(reservoir::Reservoir) = adapt(Array, reservoir)
gpu(reservoir::Reservoir) = adapt(CuArray, reservoir)

# insize(reservoir::Reservoir) = (size(reservoir.Win, 2),)
outsize(reservoir::Reservoir) = (size(reservoir.Wout, 1),)
hiddensize(reservoir::Reservoir) = (size(reservoir.Wr, 1),)

struct ReservoirState{T}
    u::T
    r::T
    z::T
end
ReservoirState(reservoir::Reservoir{T}) where {T} =
    ReservoirState(fill!(similar(reservoir.Wr, hiddensize(reservoir)), zero(T)),
                   similar(reservoir.Wr, hiddensize(reservoir)),
                   similar(reservoir.Wr, outsize(reservoir)))

state(reservoir::Reservoir, insize) = ReservoirState(reservoir)

function (reservoir::Reservoir)(state::ReservoirState, input, t, Δt; explore = true)
    # get hidden neuron firing rate
    reservoir.ξhidden!(state.r)
    state.r .+= tanh.(state.u)

    # get output neuron firing rate
    if explore
        reservoir.ξ!(state.z)
        state.z .+= reservoir.Wout * state.r
    else
        state.z .= reservoir.Wout * state.r
    end

    # update du
    du = reservoir.λ * reservoir.Wr * state.r
    du .+= reservoir.Win * input
    du .+= reservoir.Wfb * state.z
    state.u .+= Δt .* (-state.u .+ du) ./ reservoir.τ

    return state
end

struct LowPassFilter{T, S}
    τ::T
    f̄::S
end

Adapt.adapt_structure(to, lpf::LowPassFilter) = LowPassFilter(lpf.τ, adapt(to, lpf.f̄))

function (lpf::LowPassFilter)(f, Δt)
    lpf.f̄ .= (1 - Δt / lpf.τ) * lpf.f̄ .+ (Δt / lpf.τ) * f

    return lpf.f̄
end

struct RMHebb{F, S, R}
    η::F
    zlpf::S
    Plpf::R
end
function RMHebb(T, η, τ, N)
    zlpf = LowPassFilter{T, Vector{T}}(τ, zeros(T, N))
    Plpf = LowPassFilter{T, Vector{T}}(τ, zeros(T, 1))

    RMHebb{typeof(η), typeof(zlpf), typeof(Plpf)}(η, zlpf, Plpf)
end
RMHebb(reservoir::Reservoir{T}; η, τ) where {T} = RMHebb(T, η, τ, outsize(reservoir)...)

Adapt.adapt_structure(to, learner::RMHebb) =
    RMHebb(learner.η, adapt(to, learner.zlpf), adapt(to, learner.Plpf))

cpu(learner::RMHebb) = adapt(Array, learner)
gpu(learner::RMHebb) = adapt(CuArray, learner)

function (learner::RMHebb)(reservoir::Reservoir, state::ReservoirState, f, t, Δt)
    P = - @reduce sum(i) (state.z[i] - f[i])^2
    P̄ = adapt(Array, learner.Plpf(P, Δt))
    M = Int(P > only(P̄))

    z̄ = learner.zlpf(state.z, Δt)

    reservoir.Wout .+= learner.η(t) .* M .* (state.z - z̄) * transpose(state.r)

    return reservoir
end

step!(reservoir::Reservoir, state::ReservoirState, input, t, Δt; explore = true) =
    reservoir(state, input, t, Δt; explore = explore)
function step!(reservoir::Reservoir, state::ReservoirState, learner::RMHebb, input, f, t, Δt; explore = true)
    step!(reservoir, state, input, t, Δt; explore = explore)
    learner(reservoir, state, f, t, Δt)

    return state
end
