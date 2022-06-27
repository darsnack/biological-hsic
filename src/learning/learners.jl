weight_update!(layer::LIFDense, dW, opt) = Flux.Optimise.update!(opt, layer.W, dW)
weight_update!(layer::LIFConv, dW, opt) = Flux.Optimise.update!(opt, layer.W, dW)

hsiclocal(::LIFDense, zpre, zpost) = @cast _[i, j] := (1 - zpost[i]^2) * zpre[j]
function hsiclocal(layer::LIFConv, zpre, zpost)
    cdims = _cdims(layer, size(zpre))

    return NNlib.∇conv_filter(zpre, zpost, cdims)
end

hsiccombine(::Type{<:LIFDense}, ξ, β) = @cast _[i, j] := ξ[i] * β[i, j]
# hsiccombine(::Type{<:LIFConv}, ξ, β) = @cast _[i, j, m, n] := ξ[(i, j, n)] * β[i, j]

struct HSICLayer{T, S, O}
    layer::T
    learner::S
    opt::O
end

Adapt.adapt_structure(to, layer::HSICLayer) =
    HSICLayer(adapt(to, layer.layer), adapt(to, layer.learner), layer.opt)

cpu(layer::HSICLayer) = adapt(Array, layer)
gpu(layer::HSICLayer) = adapt(CuArray, layer)

outsize(layer::HSICLayer, insize) = outsize(layer.layer, insize)

state(layer::HSICLayer, insize) = (layer = state(layer.layer, insize),
                                   learner = state(layer.learner))

(layer::HSICLayer)(state, input, t, Δt) = layer.layer(state.layer, input, t, Δt)

function update!(layer::HSICLayer, state, x, y, zpre, zpost, t, Δt)
    # compute global term
    ξ = layer.learner(state.learner, x, y, zpre, zpost, t, Δt)

    # compute local term
    β = hsiclocal(layer.layer, reshape(zpre, :), reshape(zpost, :))

    # apply weight update
    @cast Δw[i, j] := ξ[i] * β[i, j]
    weight_update!(layer.layer, Δw, layer.opt)

    return layer
end

function update!(layers::LIFChain, states, x, y, zpres, zposts, t, Δt; nthreads = 1)
    if nthreads > 1
        Threads.@threads for i in 1:length(layers)
            update!(layers[i], states[i], x, y, zpres[i], zposts[i], t, Δt)
        end
    else
        foreach(layers, states, zpres, zposts) do layer, state, zpre, zpost
            update!(layer, state, x, y, zpre, zpost, t, Δt)
        end
    end
end
