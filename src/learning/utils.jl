struct GlobalError{T, S, R, Q}
    γ::T
    α::S
    ξ::R
    λ::Q
end
GlobalError{T}(bs, n; λ = 2) where {T} =
    GlobalError(zeros(T, bs), zeros(T, bs, n), zeros(T, n), λ)

Adapt.adapt_structure(to, error::GlobalError) = GlobalError(adapt(to, error.γ),
                                                            adapt(to, error.α),
                                                            adapt(to, error.ξ),
                                                            error.λ)
cpu(error::GlobalError) = adapt(Array, error)
gpu(error::GlobalError) = adapt(CuArray, error)

function (error::GlobalError)(kx, ky, kz, z; σz = estσ(z))
    bs = size(error.γ, 1)

    @cast error.γ[i] = (kx[1, i] - @reduce sum(k) kx[1, k] / bs) -
                        error.λ * (ky[1, i] - @reduce sum(k) ky[1, k] / bs)


    @cast error.α[i, k] = -2 * kz[1, i] * (z[k, 1] - z[k, i]) / σz^2
    @cast error.α[i, k] = error.α[i, k] - @reduce _[k] := sum(n) error.α[n, k] / bs

    @reduce error.ξ[k] = sum(i) error.γ[i] * error.α[i, k] / (bs - 1)^2

    return error.ξ
end

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

struct KernelMatrices{N, T<:Tuple, S<:Tuple, R<:Tuple}
    caches::S
    batches::T
    matrices::S
    sigmas::R
end

KernelMatrices{N}(caches::S,
                  batches::T,
                  matrices::S,
                  sigmas::R) where {N, T<:Tuple, S<:Tuple, R<:Tuple} =
    KernelMatrices{N, T, S, R}(caches, batches, matrices, sigmas)

function KernelMatrices(::Type{T}, sigmas, sizes, batchsize, sample_length) where T
    batches = map(sizes) do sz
        preload!(TimeBatch(T, sz...; batchsize = batchsize, sample_length = sample_length),
                 zeros(T, sz...))
    end
    caches = map(sz -> zeros(T, prod(sz), batchsize), sizes)
    matrices = ntuple(_ -> zeros(T, batchsize, batchsize), length(batches))

    KernelMatrices{length(batches)}(caches, batches, matrices, sigmas)
end

function Adapt.adapt_structure(to, kernels::KernelMatrices{N}) where N
    f = Base.Fix1(adapt, to)
    caches = map(f, kernels.caches)
    batches = map(f, kernels.batches)
    matrices = map(f, kernels.matrices)

    return KernelMatrices{N}(caches, batches, matrices, kernels.sigmas)
end

cpu(kernels::KernelMatrices) = adapt(Array, kernels)
gpu(kernels::KernelMatrices) = adapt(CuArray, kernels)

function Base.push!(kernels::KernelMatrices{N}, samples::Vararg{<:Any, N}) where N
    # @show map(x -> size(x.samples), kernels.batches)
    # @show map(size, samples)
    bs = size(kernels.matrices[1], 2)
    for i in 1:N
        # push new sample into buffer
        push!(kernels.batches[i], samples[i])
        # load latest timestep into batch cache
        @inbounds kernels.caches[i] .= reshape(view(kernels.batches[i], 0:-1:-(bs - 1), 0), :, bs)
        # compute new kernel matrix
        k_hsic!(kernels.matrices[i], kernels.caches[i]; σ = kernels.sigmas[i])
    end

    return kernels.matrices
end
