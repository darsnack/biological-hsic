struct LowPassFilter{T, S}
    τ::T
    f̄::S
end

Adapt.adapt_structure(to, lpf::LowPassFilter) = LowPassFilter(lpf.τ, adapt(to, lpf.f̄))

function (lpf::LowPassFilter)(f, Δt)
    lpf.f̄ .= (1 - Δt / lpf.τ) * lpf.f̄ .+ (Δt / lpf.τ) * f

    return lpf.f̄
end
