outsize(x, insize) = outsize(x)
outsize(chain::AbstractVector, insize) =
    foldl((sz, layer) -> push!(sz, outsize(layer, sz[end])), chain; init = Tuple[insize])
outsize(chain::Tuple, insize) = outsize([chain...], insize)

iterate_pairs(itr) = zip(itr[1:(end - 1)], itr[2:end])

function uniform_init(rng::AbstractRNG, dims::Integer...; a = -1, b = 1)
  shift = Float32(a)
  scale = Float32(b - a)

  return scale .* rand(rng, Float32, dims...) .+ shift
end
uniform_init(dims::Integer...; kwargs...) = uniform_init(rng_from_array(), dims...; kwargs...)
uniform_init(rng::AbstractRNG = Flux.rng_from_array(); init_kwargs...) =
    (dims...; kwargs...) -> uniform_init(rng, dims...; init_kwargs..., kwargs...)

function uniform_init(rng::AbstractRNG, dims::Integer...; a = -1, b = 1)
  shift = Float32(a)
  scale = Float32(b - a)

  return scale .* rand(rng, Float32, dims...) .+ shift
end
uniform_init(dims::Integer...; kwargs...) = uniform_init(rng_from_array(), dims...; kwargs...)
uniform_init(rng::AbstractRNG = Flux.rng_from_array(); init_kwargs...) =
    (dims...; kwargs...) -> uniform_init(rng, dims...; init_kwargs..., kwargs...)

function conv_chain(ksize, conv_config, param_config, act = relu)
    convs = Any[Conv(ksize, conv_config[1] => conv_config[2], act; param_config[1]...)]
    inchannels = conv_config[2]
    for (outchannels, ps) in zip(conv_config[3:end], param_config[2:end])
        if outchannels isa Integer
            push!(convs, Conv(ksize, inchannels => outchannels, act; ps...))
            inchannels = outchannels
        else
            push!(convs, outchannels)
        end
    end

    return convs
end
function dense_chain(layer_config, act = relu)
    denses = Any[Dense(layer_config[1:2]..., act)]
    insz = layer_config[2]
    for outsz in layer_config[3:end]
        if outsz isa Integer
            push!(denses, Dense(insz, outsz, act))
            insz = outsz
        else
            push!(denses, outsz)
        end
    end

    return denses
end

function lifconv_chain(::Type{T}, ksize, conv_config, param_config; kwargs...) where T
    convs = Any[LIFConv{T}(ksize, conv_config[1] => conv_config[2];
                           param_config[1]..., kwargs...)]
    inchannels = conv_config[2]
    for (outchannels, ps) in zip(conv_config[3:end], param_config[2:end])
        if outchannels isa Integer
            push!(convs, LIFConv{T}(ksize, inchannels => outchannels; ps..., kwargs...))
            inchannels = outchannels
        else
            push!(convs, outchannels)
        end
    end

    return convs
end
function lifdense_chain(::Type{T}, layer_config; kwargs...) where T
    denses = Any[LIFDense{T}(layer_config[1:2]...; kwargs...)]
    insz = layer_config[2]
    for outsz in layer_config[3:end]
        if outsz isa Integer
            push!(denses, LIFDense{T}(insz, outsz; kwargs...))
            insz = outsz
        else
            push!(denses, outsz)
        end
    end

    return denses
end

# like flatten but matching the (state, x, t, Δt) argument structure of LIF layers
lifflatten(state, x, t, Δt) = MLUtils.flatten(x)
state(::typeof(lifflatten), insize) = nothing
outsize(::typeof(lifflatten), insize) = (prod(insize[1:(end - 1)]),)
update!(::typeof(lifflatten), state, x, y, zpre, zpost, t, Δt) = nothing
