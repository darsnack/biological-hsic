struct LIFConv{T, N, M, S<:AbstractArray{T}}
    τ::T
    ξ::T
    W::S
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
end

function LIFConv{T}(ksize::NTuple{N, <:Integer}, inout::Pair{<:Integer, <:Integer};
                    stride = 1,
                    pad = 0,
                    dilation = 1,
                    τ = 50f-3,
                    init = Flux.glorot_uniform,
                    ξ = 0.05) where {T, N}
    # noise distributions
    # function ξ!(dest) # Uniform(-noise, noise)
    #     rand!(dest)
    #     @. dest = 2 * ξ * dest - ξ
    #     return dest
    # end

    # weight
    W = convert.(T, Flux.convfilter(ksize, inout; init = init))
    n = ndims(W)

    # kernel parameters
    stride = Flux.expand(Val(n - 2), stride)
    dilation = Flux.expand(Val(n - 2), dilation)
    pad = Flux.calc_padding(Conv, pad, size(W)[1:(n - 2)], dilation, stride)

    S = typeof(W)

    return LIFConv{T, n - 2, length(pad), S}(τ, ξ, W, stride, pad, dilation)
end

Adapt.adapt_structure(to, x::LIFConv) =
    LIFConv(x.τ, x.ξ, adapt(to, x.W), x.stride, x.pad, x.dilation)

cpu(x::LIFConv) = adapt(Array, x)
gpu(x::LIFConv) = adapt(CuArray, x)

_cdims(layer::LIFConv, insize) = NNlib.DenseConvDims(insize, size(layer.W);
                                                     stride = layer.stride,
                                                     dilation = layer.dilation,
                                                     padding = layer.pad)
_cdims(layer::LIFConv, insize::NTuple{3}) = _cdims(layer, (insize..., 1))

function outsize(layer::LIFConv, insize)
    cdims = _cdims(layer, insize)

    return (NNlib.output_size(cdims)..., NNlib.channels_out(cdims))
end

struct LIFConvState{T, S}
    y::T
    u::S
    r::S
end

state(layer::LIFConv{T}, insize) where T =
    LIFConvState(similar(layer.W, (outsize(layer, insize)..., 1)),
                 fill!(similar(layer.W, outsize(layer, insize)), zero(T)),
                 similar(layer.W, outsize(layer, insize)))

function (layer::LIFConv)(state::LIFConvState, input::AbstractArray{<:Any, 4}, t, Δt)
    # get neuron firing rate
    rand!(state.r)
    @. state.r = 2 * layer.ξ * state.r - layer.ξ + tanh(state.u)

    # update du
    cdims = _cdims(layer, size(input))
    NNlib.conv!(state.y, input, layer.W, cdims)
    state.u .+= Δt .* (-state.u .+ state.y) ./ layer.τ

    return state.r
end
(layer::LIFConv)(state::LIFConvState, input::AbstractArray{<:Any, 3}, t, Δt) =
    layer(state, reshape(input, size(input)..., 1), t, Δt)

step!(layer::LIFConv, state::LIFConvState, input, t, Δt) = layer(state, input, t, Δt)
