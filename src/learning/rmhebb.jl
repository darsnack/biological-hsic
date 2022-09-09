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

struct RMHebb{T}
    zlpf::T
    Plpf::T
end

function RMHebb(T, τ, Δt, N)
    zlpf = LowPassFilter{T, Vector{T}}(τ, Δt, zeros(T, N))
    Plpf = LowPassFilter{T, Vector{T}}(τ, Δt, zeros(T, 1))

    RMHebb(zlpf, Plpf)
end
RMHebb(reservoir::Recur{<:ReservoirCell{T}}; τ) where T =
    RMHebb(T, τ, reservoir.cell.Δt, size(reservoir.cell.Wout, 1))

# Plpf is always a 1-element vector so we don't move it off the CPU
Adapt.adapt_structure(to, learner::RMHebb) =
    RMHebb(adapt(to, learner.zlpf), learner.Plpf)

function (learner::RMHebb)((u, z), r, target)
    P = -sum((z .- f).^2)
    P̄ = learner.Plpf(P, Δt)
    M = Int(P > only(P̄))

    z̄ = learner.zlpf(z, Δt)

    dWout = M .* (z - z̄) * transpose(r)

    return dWout
end
