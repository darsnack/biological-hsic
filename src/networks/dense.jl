struct LIFDense{T, S<:AbstractMatrix{T}}
    τ::T
    ξ::T
    W::S
end

function LIFDense{T}(in, out; τ = 50f-3, init = Flux.glorot_uniform, ξ = 0.05) where T
    # noise distributions
    # function ξ!(dest) # Uniform(-ξ, ξ)
    #     rand!(dest)
    #     @. dest = 2 * ξ * dest - ξ
    #     return dest
    # end

    # weight
    W = convert.(T, init(out, in))

    S = typeof(W)

    return LIFDense{T, S}(τ, ξ, W)
end

Adapt.adapt_structure(to, x::LIFDense) = LIFDense(x.τ, x.ξ, adapt(to, x.W))

cpu(x::LIFDense) = adapt(Array, x)
gpu(x::LIFDense) = adapt(CuArray, x)

# insize(layer::LIFDense) = (size(layer.W, 2),)
outsize(layer::LIFDense) = (size(layer.W, 1),)

struct LIFDenseState{T}
    u::T
    r::T
end

LIFDenseState(layer::LIFDense{T}) where T =
    LIFDenseState(fill!(similar(layer.W, outsize(layer)), zero(T)),
                  similar(layer.W, outsize(layer)))

state(layer::LIFDense, insize) = LIFDenseState(layer)

function (layer::LIFDense)(state::LIFDenseState, input, t, Δt)
    # get neuron firing rate
    rand!(state.r)
    @. state.r = 2 * layer.ξ * state.r - layer.ξ + tanh(state.u)

    # update du
    state.u .+= Δt .* (-state.u .+ layer.W * input) ./ layer.τ

    return state.r
end

step!(layer::LIFDense, state::LIFDenseState, input, t, Δt) = layer(state, input, t, Δt)
