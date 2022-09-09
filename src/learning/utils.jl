struct TimeBatch{T<:CircularArrayBuffer}
    samples::T
    sample_length::Int
end

TimeBatch(::Type{T}, dims::Integer...; batchsize, sample_length) where T =
    TimeBatch(CircularArrayBuffer{T}(dims..., batchsize * sample_length), sample_length)

Adapt.adapt_structure(to, batch::TimeBatch) =
    TimeBatch(adapt(to, batch.samples), batch.sample_length)

function Base.push!(batch::TimeBatch, sample)
    # @show size(batch.samples)
    # @show size(sample)
    push!(batch.samples, sample)

    return batch
end

function preload!(batch::TimeBatch, x)
    while !CircularArrayBuffers.isfull(batch.samples)
        push!(batch.samples, x)
    end

    return batch
end

nsamples(batch::TimeBatch) = size(batch.samples)[end] ÷ batch.sample_length

_sample_index(i, t, n, len) = @. (n + i) * len + t

function Base.getindex(batch::TimeBatch{<:CircularArrayBuffer{<:Any, N}}, i, t) where N
    colons = ntuple(_ -> Colon(), N - 1)
    batch.samples[colons..., _sample_index(i, t, nsamples(batch), batch.sample_length)]
end

function Base.view(batch::TimeBatch{<:CircularArrayBuffer{<:Any, N}}, i, t) where N
    colons = ntuple(_ -> Colon(), N - 1)
    view(batch.samples, colons..., _sample_index(i, t, nsamples(batch), batch.sample_length))
end

struct KernelCache{T<:AbstractArray, S<:AbstractArray, R<:TimeBatch, U<:Real}
    matrix::T
    sample_cache::S
    time_batch::R
    sigma::U
end

function KernelCache(::Type{T}, input_size, batchsize, sample_length, sigma;
                     complete = false) where T
    time_batch = preload!(TimeBatch(T, input_size...; batchsize = batchsize,
                                                      sample_length = sample_length),
                          zeros(T, input_size))
    sample_cache = zeros(T, prod(input_size), batchsize)
    matrix = complete ? zeros(T, batchsize, batchsize) : zeros(T, batchsize)

    KernelMatrices(matrix, sample_cache, time_batch, sigma)
end

Adapt.@adapt_structure KernelCache

_update_kernel!(k::AbstractMatrix, xs::AbstractMatrix, sigma) =
    k_hsic!(k, xs; σ = sigma)
_update_kernel!(k::AbstractVector, xs::AbstractMatrix, sigma) =
    k_hsic!(k, xs[:, 1], xs; σ = sigma)

function Base.push!(kernel::KernelCache, activity) where N
    bs = size(kernel.matrix, 2)

    # push new sample into buffer
    push!(kernel.time_batch, activity)
    # load latest timestep into batch cache
    @inbounds kernel.sample_cache .= reshape(view(kernel.time_batch, 0:-1:-(bs - 1), 0), :, bs)
    # compute new kernel matrix
    _update_kernel!(kernel.matrix, kernel.sample_cache, kernel.sigma)

    return kernel.matrix
end

activity_cache(kernel::KernelCache) = kernel.sample_cache
