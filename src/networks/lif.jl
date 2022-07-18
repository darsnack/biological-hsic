struct LIFCell{T, F, S, R}
    forward::T
    activation::F
    Δt::S
    τ::S
    firing_noise::S
end

LIFCell(forward, activation = tanh; Δt = 1, τ = 1, firing_noise = 0) =
    LIFCell(forward, activation, Δt, τ, firing_noise)

function (lif::LIFCell)(u, x)
    y = lif.forward(x)
    u += (lif.Δt / lif.τ) .* (y - u)
    σ, ξ = NNlib.fast_act(activation), lif.firing_noise
    r = iszero(ξ) ? zero(u) : 2 .* ξ .+ rand!(similar(u)) .- ξ

    return u, σ.(u) .+ r
end

@functor LIFCell

LIF(args...; init_state, kwargs...) =
    Recur(LIFCell(args...; kwargs...), init_state)
