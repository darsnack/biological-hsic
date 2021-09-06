struct LIFDense{T, S<:AbstractMatrix{T}, F}
    τ::T
    W::S
    ξ!::F
end

function LIFDense{T}(in, out; τ = 50f-3, init = randn, ξ = 0.05) where T
    # noise distributions
    function ξ!(dest) # Uniform(-ξ, ξ)
        rand!(dest)
        @. dest = 2 * ξ * dest - ξ
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

cpu(x::LIFDense) = LIFDense(x.τ, adapt(Array, x.W), x.ξ!)
gpu(x::LIFDense) = LIFDense(x.τ, adapt(CuArray, x.W), x.ξ!)

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
    @. state.r += tanh(state.u)

    # update du
    state.u .+= Δt .* (-state.u .+ layer.W * input) ./ layer.τ

    return state.r
end

step!(layer::LIFDense, state::LIFDenseState, input, t, Δt) = layer(state, input, t, Δt)

struct LIFChain{T}
    layers::T
end

LIFChain(layers...) = LIFChain(layers)

Base.getindex(chain::LIFChain, i) = getindex(chain.layers, i)
Base.length(chain::LIFChain) = length(chain.layers)
Base.first(chain::LIFChain) = first(chain.layers)
Base.last(chain::LIFChain) = last(chain.layers)
Base.iterate(chain::LIFChain) = iterate(chain.layers)
Base.iterate(chain::LIFChain, state) = iterate(chain.layers, state)
Base.lastindex(chain::LIFChain) = lastindex(chain.layers)

nin(chain::LIFChain) = nin(first(chain))
nout(chain::LIFChain) = nout(last(chain))

cpu(x::LIFChain) = LIFChain(cpu.(x.layers)...)
gpu(x::LIFChain) = LIFChain(gpu.(x.layers)...)

state(chain::LIFChain) = Tuple([state(layer) for layer in chain])

function _applychain(layers::Tuple, states, x, t, Δt)
    y = first(layers)(first(states), x, t, Δt)

    return (y, _applychain(Base.tail(layers), Base.tail(states), y, t, Δt)...)
end
_applychain(layers::Tuple{}, states, x, t, Δt) = ()

(chain::LIFChain)(states, input, t, Δt) = _applychain(chain.layers, states, input, t, Δt)

step!(layer::LIFChain, states, input, t, Δt) = layer(states, input, t, Δt)
