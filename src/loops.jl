function initialize_grads(ps::Zygote.Params)
    cache = IdDict{Any, Any}()
    for p in ps
        cache[p] = zero(p)
    end

    return Zygote.Grads(cache, ps)
end

struct RateEncoded{T<:Phase, S<:Real} <: Phase
    phase::T
    Δt::S
    Δtsample::S
end

function Base.show(io::IO, phase::RateEncoded)
    print(io, "RateEncoded(")
    print(io, phase.phase)
    print(io, ", ")
    print(io, phase.Δt)
    print(io, ", ")
    print(io, phase.Δtsample)
    print(io, ")")
end

FluxTraining.phasedataiter(phase::RateEncoded) =
    FluxTraining.phasedataiter(phase.phase)

function FluxTraining.step!(learner, phase::RateEncoded, batch)
    for _ in 0:Δt:phase.Δtsample
        FluxTraining.step!(learner, phase.phase, batch)
    end
end

struct RMHebbTraining{T<:RMHebb} <: AbstractTrainingPhase
    rmhebb::T
end

Base.show(io::IO, ::RMHebbTraining) = print(io, "RMHebbTraining(...)")

function FluxTraining.step!(learner, phase::RMHebbTraining, sample)
    x, y = sample
    FluxTraining.runstep(learner, phase, (xs = x, ys = y)) do handle, state
        state.grads = initialize_grads(learner.params)
        handle(FluxTraining.LossBegin())
        learner.model(state.xs)
        handle(FluxTraining.BackwardBegin())
        dWout = phase.rmhebb(learner.model.state, state.ys)
        state.grads[learner.model.cell.Wout] .+= dWout
        state.ŷs = phase.rmhebb.zlpf.f̄
        state.loss = learner.lossfn(state.ŷs, state.ys)
        handle(FluxTraining.BackwardEnd())
        Flux.Optimise.update!(learner.optimizer, learner.model.cell.Wout, dWout)
    end
end

struct RMHebbValidation{T<:LowPassFilter} <: AbstractValidationPhase
    lpf::T
end

function RMHebbValidation(rmhebb::RMHebb)
    lpf = LowPassFilter(rmhebb.zlpf.τ, rmhebb.zlpf.Δt, similar(rmhebb.zlpf.f̄))
    lpf.f̄ .= zero(lpf.f̄)

    RMHebbValidation(lpf)
end

Base.show(io::IO, ::RMHebbValidation) = print(io, "RMHebbValidation(...)")

function FluxTraining.step!(learner, phase::RMHebbValidation, sample)
    x, y = sample
    FluxTraining.runstep(learner, phase, (xs = x, ys = y)) do handle, state
        ŷs = learner.model(state.xs)
        state.ŷs = phase.lpf(ŷs)
        state.loss = learner.lossfn(state.ŷs, state.ys)
    end
end

# function run!(step!::F, data, args...;
#               nepochs = 1, bs = 1, shuffle = true, progress = nothing) where F
#     progressname, progressrate = isnothing(progress) ? ("", 0) : progress
#     progressi = 0
#     n = nepochs * floor(Int, numobs(data) / bs)
#     @withprogress name=progressname begin
#         for epoch in 1:nepochs
#             @info "($progressname): Starting epoch $epoch ..."

#             dataloader = shuffle ? shuffleobs(data) : data
#             for (i, (x, y)) in enumerate(eachobs(dataloader; batchsize = bs))
#                 step!(i, x, y, args...)
#                 progressi += 1

#                 !isnothing(progress) && (mod(progressi - 1, progressrate) == 0) &&
#                     @logprogress progressi / n
#             end
#         end
#     end
# end

# function run!(step!::F, ts::AbstractVector, args...) where F
#     for t in ts
#         step!(t, args...)
#     end
# end

# function predict(net, xs, ys; Δt, Δtsample, progress = "TEST")
#     ŷs = similar(ys, Float32)
#     net = net |> cpu
#     net_state = state(net, size(getobs(xs, 1)))
#     run!((xs, ys), net, net_state; shuffle = false, progress = progress) do i, x, _, net, net_state
#         z = mean(last(net(net_state, x, t, Δt)) for t in 0:Δt:Δtsample)
#         ŷs[:, i] .= z
#     end

#     return ŷs
# end

# function fitdecoder!(decoder, ŷs, ys; nepochs = 1000, batchsize = 32, opt = Momentum())
#     loss(y, ŷ) = let decoder = decoder
#         Flux.Losses.logitcrossentropy(decoder(ŷ), y)
#     end
#     ps = Flux.params(decoder)
#     for _ in 1:nepochs
#         for d in eachobs((ys, ŷs); batchsize = batchsize)
#             gs = gradient(ps) do
#                 loss(d...)
#             end
#             Flux.update!(opt, ps, gs)
#         end
#     end

#     return decoder
# end
# fitdecoder(ŷs, ys; kwargs...) = fitdecoder!(Dense(size(ys, 1), size(ŷs, 1)), ŷs, ys; kwargs...)
