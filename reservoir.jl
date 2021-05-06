struct Reservoir{T, S<:AbstractMatrix{T}, F, G}
    Wr::S
    Win::S
    Wfb::S
    Wout::S
    ξhidden::F
    ξ::G
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
    b = zeros(T, nout)

    S = typeof(Wr)
    F = typeof(Dξhidden)
    G = typeof(Dξ)

    Reservoir{T, S, F, G}(Wr, Win, Wfb, Wout, b, Dξhidden, Dξ, τ, λ)
end