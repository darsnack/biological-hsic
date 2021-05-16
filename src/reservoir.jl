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
    Dr = Normal(0, sqrt(1 / (sparsity * nhidden))) # weight distribution of recurrent connection
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

struct ReservoirState{T}
    u::T
    r::T
    z::T
end
ReservoirState(reservoir::Reservoir{T}) where T =
    ReservoirState(fill!(similar(reservoir.Wr, nhidden(reservoir)), zero(T)),
                   similar(reservoir.Wr, nhidden(reservoir)),
                   similar(reservoir.Wr, nout(reservoir)))

function (reservoir::Reservoir)(state::ReservoirState, input, t, Δt; explore = true)
    # get hidden neuron firing rate
    reservoir.ξhidden!(state.r)
    state.r .+= reservoir.f.(state.u)

    # get output neuron firing rate
    if explore
        reservoir.ξ!(state.z)
        state.z .+= reservoir.Wout * state.r
    else
        state.z .= reservoir.Wout * state.r
    end

    # update du
    state.u .+= Δt .* (-state.u .+ reservoir.λ * reservoir.Wr * state.r .+
                                   reservoir.Win * input(t) .+
                                   reservoir.Wfb * state.z) ./ reservoir.τ

    return state
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
RMHebb(reservoir::Reservoir{T}; η, τ) where {T} = RMHebb(T, η, τ, nout(reservoir))

cpu(learner::RMHebb) = RMHebb(learner.η,
                              LowPassFilter(learner.zlpf.τ, adapt(Array, learner.zlpf.f̄)),
                              LowPassFilter(learner.Plpf.τ, adapt(Array, learner.Plpf.f̄)))
gpu(learner::RMHebb) = RMHebb(learner.η,
                              LowPassFilter(learner.zlpf.τ, adapt(CuArray, learner.zlpf.f̄)),
                              LowPassFilter(learner.Plpf.τ, adapt(CuArray, learner.Plpf.f̄)))

function (learner::RMHebb)(reservoir::Reservoir, state::ReservoirState, f, t, Δt)
    P = -sum((state.z .- f(t)).^2)
    P̄ = learner.Plpf(P, Δt)
    M = adapt(typeof(state.z), @. P > P̄)

    z̄ = learner.zlpf(state.z, Δt)

    reservoir.Wout .+= learner.η(t) * (state.z .- z̄) .* M * transpose(state.r)

    return reservoir
end

step!(reservoir::Reservoir, state::ReservoirState, input, t, Δt; explore = true) =
    reservoir(state, input, t, Δt; explore = explore)
function step!(reservoir::Reservoir, state::ReservoirState, learner::RMHebb, input, f, t, Δt; explore = true)
    step!(reservoir, state, input, t, Δt; explore = explore)
    learner(reservoir, state, f, t, Δt)

    return state
end