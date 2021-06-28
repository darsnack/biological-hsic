cpu(x::AbstractArray) = x
cpu(x::CuArray) = adapt(Array, x)
gpu(x::AbstractArray) = adapt(CuArray, x)
gpu(x::CuArray) = x

concatenate(X, Y, Z) = vcat(X, Y, Z)
unconcatenate(I) = I[1:Nx], I[(Nx + 1):(Nx + Ny)], I[(Nx + Ny + 1):end]

_zero(::Type{T}, dims...) where {T<:AbstractArray} = adapt(T, zeros(eltype(T), dims...))

trange(start, Δt, span) = start:Δt:(start + span - Δt)

# inplace push! for CircularBuffer
@inline function push_inplace!(cb::CircularBuffer, data)
    # if full, increment and overwrite, otherwise push
    if cb.length == cb.capacity
        cb.first = (cb.first == cb.capacity ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    @inbounds cb.buffer[DataStructures._buffer_index(cb, cb.length)] .= data
    return cb
end

## plot utils

"""
    classificationplot!(axis, X, Y)
Add scatter plot to `axis` with color coded class labels.
Returns the array of plots added to the axis (one for each class).
# Arguments:
- `axis::LAxis`: a MakieLayout.jl `LAxis` object
- `X`: a 2xN matrix of input data (N is the number of samples)
- `Y`: a CxN matrix of output data (C is the number of classes)
- `color`: the colors to use when `size(Y, 1) == 1` (unused for multiclass)
"""
function classificationplot!(axis, X, Y; color = [:blue, :red])
    plts = []
    D = size(Y, 1)
    if D == 1
        Y = vec(Y)
        push!(plts, scatter!(axis, X[1, Y .== 1], X[2, Y .== 1]; color = color[1], marker = :circle, markersize = 10px))
        push!(plts, scatter!(axis, X[1, Y .== 0], X[2, Y .== 0]; color = color[2], marker = :circle, markersize = 10px))
    else
        for i in 1:D
            idx = getindex.(vec(argmax(Y; dims = 1)), 1)
            push!(plts, scatter!(axis, X[1, idx .== i], X[2, idx .== i]; color = rand(RGB), marker = :circle, markersize = 10px))
        end
    end

    return plts
end
classificationplot!(axis, data::Tuple; kwargs...) = classificationplot!(axis, data...; kwargs...)
classificationplot!(axis, data::NamedTuple{(:x, :y)}; kwargs...) =
    classificationplot!(axis, data...; kwargs...)
function classificationplot!(axis, data::AbstractArray{<:Tuple, 1}; kwargs...)
    plts = classificationplot!(axis, data[1][1], data[1][2]; kwargs...)
    for (x, y) in data[2:end]
        classificationplot!(axis, x, y; kwargs...)
    end

    return plts
end
classificationplot!(axis, data::AbstractArray{<:NamedTuple{(:x, :y)}, 1}; kwargs...) =
    map(s -> classificationplot!(axis, s.x, s.y; kwargs...), data)

function decisionboundary!(axis, W; ϕ = identity, f = x -> x > 0 ? 1 : 0, npts = 200)
    (size(W, 2) != 2) && error("Cannot visualize more than 2D features!")
    xmin = -1
    xmax = 1
    ymin = -1
    ymax = 1
    Δx = (xmax - xmin) / npts
    Δy = (ymax - ymin) / npts
    x = range(xmin, step = Δx, length = npts)
    y = range(ymin, step = Δy, length = npts)
    features = vcat(repeat(reshape(x, 1, :), inner = (1, npts)), repeat(reshape(y, 1, :), outer = (1, npts)))
    Y = f.(W * mapreduce(ϕ, hcat, eachcol(features)))
    Y = permutedims(reshape(Y, npts, npts))

    if all(Y .== first(Y))
        @warn "Not drawing decision boundary because there is only one resulting class"
        return nothing
    end

    return contour!(axis, vec(x), vec(y), Y; linewidth = 2, levels = 1)
end
