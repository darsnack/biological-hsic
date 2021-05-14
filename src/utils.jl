cpu(x::AbstractArray) = x
cpu(x::CuArray) = adapt(Array, x)
gpu(x::AbstractArray) = adapt(CuArray, x)
gpu(x::CuArray) = x