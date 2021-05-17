cpu(x::AbstractArray) = x
cpu(x::CuArray) = adapt(Array, x)
gpu(x::AbstractArray) = adapt(CuArray, x)
gpu(x::CuArray) = x

concatenate(X, Y, Z) = vcat(X, Y, Z)
unconcatenate(I) = I[1:Nx], I[(Nx + 1):(Nx + Ny)], I[(Nx + Ny + 1):end]