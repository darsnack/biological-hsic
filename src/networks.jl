struct LIFDense{T, S<:AbstractMatrix{T}, F}
    τ::T
    W::S
    ξ!::F
end

function LIFDense{T}(in, out; τ = 50f-3, f = tanh, init = randn) where T
    # noise distributions
    function ξ!(dest) # Uniform(-0.05, 0.05)
        rand!(dest)
        @. dest = 0.1 * dest - 0.05
        return dest
    end

    # weight
    W = convert.(T, init(out, in))

    S = typeof(W)
    F = typeof(ξ!)

    return LIFDense{T, S, F}(τ, W, ξ!)
end

nin(layer::LIFDense) = size(layer.W, 2)
nout(layer::LIFDense) = size(layer.W, 1)

cpu(x::LIFDense) = LIFDense(x.τ, adapt(Array, x.W), layer.ξ!)
gpu(x::LIFDense) = LIFDense(x.τ, adapt(CuArray, x.W), layer.ξ!)

struct LIFDenseState{T}
    u::T
    r::T
end

LIFDenseState(layer::LIFDense{T}) where {T} =
    LIFDenseState(fill!(similar(layer.W, nout(layer)), zero(T)), similar(layer.W, nout(layer)))

state(layer::LIFDense) = LIFDenseState(layer)

function (layer::LIFDense)(state::LIFDenseState, input, t, Δt)
    # get neuron firing rate
    layer.ξ!(state.r)
    state.r .+= tanh.(state.u)

    # update du
    state.u .+= Δt .* (-state.u .+ layer.W * input) ./ layer.τ

    return state.r
end

step!(layer::LIFDense, state::LIFDenseState, input, t, Δt) = layer(state, input, t, Δt)
