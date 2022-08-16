struct LowPassFilter{T, S}
    τ::T
    Δt::T
    f̄::S
end

Adapt.adapt_structure(to, lpf::LowPassFilter) =
    LowPassFilter(lpf.τ, lpf.Δt, adapt(to, lpf.f̄))

function (lpf::LowPassFilter)(f)
    lpf.f̄ .= (1 - lpf.Δt / lpf.τ) * lpf.f̄ .+ (lpf.Δt / lpf.τ) * f

    return lpf.f̄
end

struct RMHebb{T, S}
    η::T
    zlpf::S
    Plpf::S
end
function RMHebb(T, η, τ, Δt, N)
    zlpf = LowPassFilter{T, Vector{T}}(τ, Δt, zeros(T, N))
    Plpf = LowPassFilter{T, Vector{T}}(τ, Δt, zeros(T, 1))

    RMHebb(η, zlpf, Plpf)
end
RMHebb(reservoir::Reservoir{T}; η, τ) where {T} =
    RMHebb(T, η, τ, reservoir.Δt, size(reservoir.Wout, 1))

Adapt.adapt_structure(to, learner::RMHebb) =
    RMHebb(learner.η, adapt(to, learner.zlpf), adapt(to, learner.Plpf))

# cpu(learner::RMHebb) = adapt(Array, learner)
# gpu(learner::RMHebb) = adapt(CuArray, learner)

function (learner::RMHebb)(reservoir::Reservoir, f, t, Δt)
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
