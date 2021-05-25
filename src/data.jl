"""
    generatedata(D, N; Dx = nothing, ϕ = [identity], f = [x -> x > 0 ? 1 : 0], W = [nothing],
                       usebias = false, batchsize = N)
Generate labeled synthetic data by doing the following:
 1. Generate `N` data points in `D` dimensions
    (drawing from `Dx` if specified, else `Unif([0, 1])`).
 2. For each pair, `(ϕ[i], f[i])` obtain the feature vector as `ϕ[i](x)`.
 3. If `W[i]` is `nothing`, then generate a random decision boundary with weight matrix `W[i]`.
    Otherwise, use `W[i]` directly.
 4. Label the data point as `f[i](W[i] * ϕ[i](x))`
    (`f[i]` maps each weighted feature vector to 0 or 1).
 5. If `length(ϕ) == length(f) > 1`, multiple binary class vectors are generated.
    Remap these vectors into one-hot vectors. Single class outputs are kept as a scalars.
 6. Return the `data` as an array of batches where each batch is a tuple, `(X, Y)`.
    Also return the decision boundaries `W`.
"""
function generatedata(D, N; Dx = nothing, ϕ = [identity], f = [x -> x > 0 ? 1 : 0],
                            W = fill(nothing, length(ϕ)),
                            usebias = false)
    Dout = length(f)
    X = isnothing(Dx) ? rand(D, N) : rand(Dx, N)
    X = usebias ? stackbias(X) : X

    Ŵ = Array{Matrix}(undef, Dout)
    Y = zeros(Dout, N)
    for (i, (ϕi, fi)) in enumerate(zip(ϕ, f))
        ϕX = mapreduce(ϕi, hcat, eachcol(X))

        Ŵ[i] = isnothing(W[i]) ? rand(1, size(ϕX, 1)) : W[i]
        Y[i, :] = fi.(Ŵ[i] * ϕX)
    end

    # handle multiple classes
    if Dout > 1
        y = 1 .- sum(Y; dims = 1)
        Y = vcat(Y, y)
    end

    return (X, Y), Ŵ
end

stackbias(X) = vcat(X, ones(eltype(X), 1, size(X, 2)))
stackbias(data::Tuple) = (stackbias(data[1]), data[2])

struct RateEncoder{T, S}
    data::T
    Δt::S
end

Base.length(encoder::RateEncoder) = size(encoder.data, 2)
nout(encoder::RateEncoder) = size(encoder.data, 1)

Base.eltype(encoder::RateEncoder) = typeof(encoder(0))

cpu(encoder::RateEncoder) = RateEncoder(cpu(encoder.data), encoder.Δt)
gpu(encoder::RateEncoder) = RateEncoder(gpu(encoder.data), encoder.Δt)

function (encoder::RateEncoder)(t)
    Nsamples = length(encoder)
    i = (t < 0) ? 1 : (Int(round(t / encoder.Δt)) % Nsamples) + 1

    return encoder.data[:, i]
end
