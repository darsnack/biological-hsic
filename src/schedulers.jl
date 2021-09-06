struct Interpolator{T, S<:Real}
    schedule::T
    rate::S
end

(interpolator::Interpolator)(t) = interpolator.schedule(cld(t, interpolator.rate))
Base.iterate(interpolator::Interpolator, state = 1) = (interpolator(state), state + 1)
Base.IteratorEltype(::Type{<:Interpolator{T}}) where T = Base.IteratorEltype(T)
Base.IteratorSize(::Type{<:Interpolator{T}}) where T = Base.IteratorSize(T)
Base.eltype(::Type{<:Interpolator{T}}) where T = eltype(T)
