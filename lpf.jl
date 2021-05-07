struct LowPassFilter{T, S}
    τ::T
    f̄::S
end

function (lpf::LowPassFilter)(f, Δt)
    lpf.f̄ .= (1 - Δt / lpf.τ) * lpf.f̄ + (Δt / τ) * f

    return lpf.f̄
end