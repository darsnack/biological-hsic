struct ReservoirCell{T, S<:AbstractMatrix{T}}
    Wr::S
    Win::S
    Wfb::S
    Wout::S
    hidden_noise::T
    output_noise::T
    τ::T
    λ::T
    Δt::T
end
function ReservoirCell{T}(nin, nout, nhidden;
                          τ = 10e-3, λ = 1.2, Δt = 1,
                          sparsity = 0.1, hidden_noise = 0.05, output_noise = 0.5) where T
    # initial values
    Wr = convert.(T, Flux.sparse_init(nhidden, nhidden;
                                      sparsity = sparsity,
                                      std = sqrt(1 / sparsity * nhidden)))
    Win = convert.(T, uniform_init(nhidden, nin))
    Wfb = convert.(T, uniform_init(nhidden, nout))
    Wout = zeros(T, nout, nhidden)

    S = typeof(Wr)

    return ReservoirCell{T, S}(Wr, Win, Wfb, Wout, hidden_noise, output_noise, τ, λ, Δt)
end
ReservoirCell{T}(inout::Pair, nhidden; kwargs...) where {T} =
    ReservoirCell{T}(inout[1], inout[2], nhidden; kwargs...)

# Adapt.adapt_structure(to, reservoir::Reservoir) = Reservoir(adapt(to, reservoir.Wr),
#                                                             adapt(to, reservoir.Win),
#                                                             adapt(to, reservoir.Wfb),
#                                                             adapt(to, reservoir.Wout),
#                                                             reservoir.ξhidden!,
#                                                             reservoir.ξ!,
#                                                             reservoir.τ,
#                                                             reservoir.λ)

# cpu(reservoir::Reservoir) = adapt(Array, reservoir)
# gpu(reservoir::Reservoir) = adapt(CuArray, reservoir)

# # insize(reservoir::Reservoir) = (size(reservoir.Win, 2),)
# outsize(reservoir::Reservoir) = (size(reservoir.Wout, 1),)
# hiddensize(reservoir::Reservoir) = (size(reservoir.Wr, 1),)

# struct ReservoirState{T}
#     u::T
#     z::T
# end
# ReservoirState(reservoir::Reservoir{T}) where {T} =
#     ReservoirState(fill!(similar(reservoir.Wr, hiddensize(reservoir)), zero(T)),
#                    similar(reservoir.Wr, outsize(reservoir)))

# state(reservoir::Reservoir, insize) = ReservoirState(reservoir)

function (reservoir::ReservoirCell)(u, x)
    ξh, ξo = reservoir.hidden_noise, reservoir.output_noise

    # get hidden neuron firing rate
    r = iszero(ξh) ? zero(u) : 2 .* ξh .+ rand!(similar(u)) .- ξh
    r .+= tanh.(state.u)

    # get output neuron firing rate
    # if explore
    #     reservoir.ξ!(state.z)
    #     state.z .+= reservoir.Wout * state.r
    # else
    #     state.z .= reservoir.Wout * state.r
    # end
    z = iszero(ξo) ? zero(u) : 2 .* ξo .+ rand!(similar(u)) .- ξo
    z .+= reservoir.Wout * r

    # update du
    du = reservoir.λ * reservoir.Wr * r
    du .+= reservoir.Win * x
    du .+= reservoir.Wfb * z
    u .+= Δt .* (-u .+ du) ./ reservoir.τ

    return u, r
end

@functor ReservoirCell
Flux.trainable(reservoir::ReservoirCell) = (Wout = reservoir.Wout,)

Reservoir(args...; init_state, kwargs...) =
    Recur(ReservoirCell(args...; kwargs...), init_state)
