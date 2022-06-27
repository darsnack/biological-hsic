struct LIFChain{T<:Tuple}
    layers::T
end

LIFChain(layers...) = LIFChain(layers)
LIFChain(layers::AbstractVector) = LIFChain(layers...)

Base.getindex(chain::LIFChain, i) = getindex(chain.layers, i)
Base.length(chain::LIFChain) = length(chain.layers)
Base.first(chain::LIFChain) = first(chain.layers)
Base.last(chain::LIFChain) = last(chain.layers)
Base.iterate(chain::LIFChain) = iterate(chain.layers)
Base.iterate(chain::LIFChain, state) = iterate(chain.layers, state)
Base.lastindex(chain::LIFChain) = lastindex(chain.layers)

cpu(x::LIFChain) = LIFChain(map(cpu, x.layers))
gpu(x::LIFChain) = LIFChain(map(gpu, x.layers))

outsize(chain::LIFChain, insize) = outsize(chain.layers, insize)[end]

state(chain::LIFChain, insize) = Tuple(map(state, chain.layers, outsize(chain.layers, insize)))

function _applychain(layers::Tuple, states, x, t, Δt)
    y = first(layers)(first(states), x, t, Δt)

    return (y, _applychain(Base.tail(layers), Base.tail(states), y, t, Δt)...)
end
_applychain(layers::Tuple{}, states, x, t, Δt) = ()

(chain::LIFChain{<:Tuple})(states, input, t, Δt) =
    (input, _applychain(chain.layers, states, input, t, Δt)...)
(chain::LIFChain)(states, input, t, Δt) =
    foldl(zip(states, chain.layers); init = [input]) do zs, (state, layer)
        push!!(zs, layer(state, zs[end], t, Δt))
    end

step!(layer::LIFChain, states, input, t, Δt) = layer(states, input, t, Δt)
