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

struct RMHebb{T, S, R}
    zlpf::T
    Plpf::S
    gain::R
end

function RMHebb(T, τ, Δt, N; gain = 1)
    zlpf = LowPassFilter{T, Vector{T}}(τ, Δt, zeros(T, N))
    Plpf = LowPassFilter{T, Vector{T}}(τ, Δt, zeros(T, 1))

    RMHebb(zlpf, Plpf, gain)
end
RMHebb(reservoir::Recur{<:ReservoirCell{T}}; τ, gain = 1) where T =
    RMHebb(T, τ, reservoir.cell.Δt, size(reservoir.cell.Wout, 1); gain = gain)

# Plpf is always a 1-element vector so we don't move it off the CPU
Adapt.adapt_structure(to, learner::RMHebb) =
    RMHebb(adapt(to, learner.zlpf), learner.Plpf, learner.gain)

function (learner::RMHebb)((u, r, z), target)
    P = -sum((z .- learner.gain .* target).^2)
    P̄ = learner.Plpf(P)
    M = Int(P > only(P̄))

    z̄ = learner.zlpf(z)

    dWout = -M .* (z .- z̄) * transpose(r)

    return dWout
end
