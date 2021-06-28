using Adapt
using LoopVectorization
using CUDA
CUDA.allowscalar(false)

using ProgressLogging
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using Distributions, Random
using DataStructures
using LinearAlgebra
using TensorCast
using Flux
using MLDataPattern, MLDatasets

include("../src/utils.jl")
# include("../src/circularbuffer.jl")
include("../src/hsic.jl")
include("../src/lpf.jl")
include("../src/reservoir.jl")
include("../src/networks.jl")
include("../src/data.jl")
include("../src/learning.jl")
include("../src/loops.jl")

function dense_test(data, target;
                    τff = 5f-3, # LIF time constant for FF network
                    τr = 50f-3, # LIF time constant for reservoirs
                    λ = 1.7, # chaotic level
                    γ = 2f0, # HSIC balance parameter
                    η0 = 1f-3, # initial hsic learning rate
                    ηr = 1f-4, # initial reservoir learning rate
                    Nhidden = 1000, # size of reservoir
                    noise = 5f-1,
                    τavg = 10f-3, # signal smoothing constant
                    Δt = 1f-3, # simulation time step
                    Nsamples = 100, # number of data samples
                    Δtsample = 50f-3, # time to present each data sample
                    bs = 15, # effective batch size
                    nepochs = 15, # number of epochs
                    Tinit = 50f0, # warmup time
                    Tpre = 500f0) # pretraining time

    η(t; ηi, toffset = zero(t), rate = 25f0) =
        (t > toffset) ? ηi / (1 + (t - toffset) / rate) : zero(Float32)
    H = I - fill(1 / bs, bs, bs) |> target # centering matrix
    hsic_loss(Kx, Kz) = tr(Kx * H * Kz * H) / (bs - 1)^2

    ## NETWORK SETUP

    Din = size(data[1], 1)
    Dout = size(data[2], 1)

    net = LIFDense{Float32}(Din, Dout; τ = τff) |> target
    net_state = state(net)

    reservoir = Reservoir{Float32}(Din + nout(net) + Dout => nout(net), Nhidden;
                                   τ = τr, λ = λ, noiseout = noise) |> target
    reservoir_learner = RMHebb(reservoir; η = t -> η(t; ηi = ηr, toffset = Tinit), τ = τavg) |> target

    learner = HSIC(data..., net, net_state, reservoir, reservoir_learner, bs;
                   γ = γ, nbuffer = Int(cld(Δtsample, Δt))) |> target
    learner_state = state(learner)

    ## RECORDING SETUP

    recording = (t = Float32[], lossx = Float32[], lossy = Float32[], w1 = Float32[], w2 = Float32[])
    function record!(recording, t, net, learner)
        lossx = hsic_loss(learner.Kx, learner.Kz)
        lossy = hsic_loss(learner.Ky, learner.Kz)
        w = net.W |> cpu

        push!(recording.t, t)
        push!(recording.lossx, lossx)
        push!(recording.lossy, lossy)
        push!(recording.w1, w[1])
        push!(recording.w2, w[2])
    end
    record_rate = Int(cld(Δtsample, Δt))

    ## WARMUP

    @info "Starting warmup..."
    @withprogress name="WARMUP" begin
        run!(0:Δt:Tinit,
             data, net, learner, net_state, learner_state) do t, data, net, learner, net_state, learner_state
            # evaluate model
            x, y = rateencode(data, t; Δt = Δtsample) .|> target
            z = net(net_state, x, t, Δt)

            # get weight update
            learner(learner_state, x, y, x, copy(z), t, Δt)

            # record values
            (cld(t, Δt) % record_rate == 0) && record!(recording, t, net, learner)

            @logprogress t / Tinit
        end
    end
    tcurrent = Tinit

    ## PRETRAINING

    @info "Starting pre-training..."
    @withprogress name="PRETRAIN" begin
        run!(tcurrent:Δt:(tcurrent + Tpre),
             data, net, learner, net_state, learner_state) do t, data, net, learner, net_state, learner_state
            # evaluate model
            x, y = rateencode(data, t; Δt = Δtsample) .|> target
            z = net(net_state, x, t, Δt)

            # get weight update
            learner(learner_state, x, y, x, copy(z), t, Δt)

            # record values
            (cld(t, Δt) % record_rate == 0) && record!(recording, t, net, learner)

            @logprogress (t - tcurrent) / Tpre
        end
    end
    tcurrent += Tpre

    ## TRAINING

    @info "Starting training..."
    run!(data, net, learner, net_state, learner_state;
         nepochs = nepochs, progress = "TRAIN") do _, x, y, net, learner, net_state, learner_state
        for t in tcurrent:Δt:(tcurrent + Δtsample)
            # evaluate model
            x, y = target(x), target(y)
            z = net(net_state, x, t, Δt)

            # get weight update
            update!(net, learner, learner_state, x, y, x, copy(z), t, Δt) do t
                η(t; ηi = η0, toffset = tcurrent, rate = 30f0)
            end

            # record values
            (cld(t, Δt) % record_rate == 0) && record!(recording, t, net, learner)
        end
        # update time
        tcurrent += Δtsample
    end

    ## POST-TRAIN

    @info "Starting testing..."
    Ŷs = similar(data[2])
    net = net |> cpu
    net_state = state(net)
    run!(data, net, net_state; shuffle = false, progress = "TEST") do i, x, _, net, net_state
        z = mean(net(net_state, x, t, Δt) for t in 0:Δt:Δtsample)
        Ŷs[:, i] .= z
    end

    @info "Starting decoding..."
    decoder = Dense(Dout, Dout)
    opt = Momentum(1e-4)
    loss(y, ŷ) = Flux.Losses.logitbinarycrossentropy(y, decoder(ŷ))
    @progress for _ in 1:1000
        Flux.train!(loss, Flux.params(decoder), [(data[2], Ŷs)], opt)
    end

    acc = mean(round.(Flux.sigmoid.(decoder(Ŷs))) .== data[2])
    @info "Accuracy = $(acc * 100)%"

    return recording
end

function chain_test(data, layer_config, target;
                    τff = 5f-3, # LIF time constant for FF network
                    τr = 50f-3, # LIF time constant for reservoirs
                    λ = 1.7, # chaotic level
                    γ = 2f0, # HSIC balance parameter
                    η0 = 1f-3, # initial hsic learning rate
                    ηr = 1f-4, # initial reservoir learning rate
                    Nhidden = 1000, # size of reservoir
                    noise = 5f-1,
                    τavg = 10f-3, # signal smoothing constant
                    Δt = 1f-3, # simulation time step
                    Δtsample = 50f-3, # time to present each data sample
                    bs = 15, # effective batch size
                    nepochs = 15, # number of epochs
                    Tinit = 50f0, # warmup time
                    Tpre = 500f0) # pretraining time

    η(t; ηi, toffset = zero(t), rate = 25f0) =
        (t > toffset) ? ηi / (1 + (t - toffset) / rate) : zero(Float32)
    H = I - fill(1 / bs, bs, bs) |> target # centering matrix
    hsic_loss(Kx, Kz) = tr(Kx * H * Kz * H) / (bs - 1)^2

    ## NETWORK SETUP

    Din = size(data[1], 1)
    Dout = size(data[2], 1)

    net = LIFChain([LIFDense{Float32}(in, out; τ = τff) for (in, out) in layer_config]) |> target
    net_state = state(net)

    learners = []
    for (layer, layer_state) in zip(net, net_state)
        reservoir = Reservoir{Float32}(Din + nout(layer) + Dout => nout(layer),
                                       Nhidden; τ = τr, λ = λ, noiseout = noise) |> target
        reservoir_learner = RMHebb(reservoir;
                                   η = t -> η(t; ηi = ηr, toffset = Tinit), τ = τavg) |> target

        learner = HSIC(data...,
                       layer,
                       layer_state,
                       reservoir,
                       reservoir_learner,
                       bs; γ = γ, nbuffer = Int(cld(Δtsample, Δt))) |> target

        push!(learners, learner)
    end
    learner_states = state.(learners)

    ## RECORDING SETUP

    recording = (t = Float32[],
                 lossxs = [Float32[] for _ in net],
                 lossys = [Float32[] for _ in net])
    function record!(recording, t, learners)
        push!(recording.t, t)
        for (i, learner) in enumerate(learners)
            lossx = hsic_loss(learner.Kx, learner.Kz)
            lossy = hsic_loss(learner.Ky, learner.Kz)
            push!(recording.lossxs[i], lossx)
            push!(recording.lossys[i], lossy)
        end
    end
    record_rate = Int(cld(Δtsample, Δt))

    ## WARMUP

    @info "Starting warmup..."
    @withprogress name="WARMUP" begin
        run!(0:Δt:Tinit,
             data, net, learners, net_state, learner_states) do t, data, net, learners, net_state, learner_states
            # evaluate model
            x, y = rateencode(data, t; Δt = Δtsample) .|> target
            zs = net(net_state, x, t, Δt)

            # get weight update
            zpres = (x, Base.front(zs)...)
            update!(t -> 0, net, learners, learner_states, x, y, copy.(zpres), copy.(zs), t, Δt)

            # record values
            (cld(t, Δt) % record_rate == 0) && record!(recording, t, learners)

            @logprogress t / Tinit
        end
    end
    tcurrent = Tinit

    ## PRETRAINING

    @info "Starting pre-training..."
    @withprogress name="PRETRAIN" begin
        run!(tcurrent:Δt:(tcurrent + Tpre),
             data, net, learners, net_state, learner_states) do t, data, net, learners, net_state, learner_states
            # evaluate model
            x, y = rateencode(data, t; Δt = Δtsample) .|> target
            zs = net(net_state, x, t, Δt)

            # get weight update
            zpres = (x, Base.front(zs)...)
            update!(t -> 0, net, learners, learner_states, x, y, copy.(zpres), copy.(zs), t, Δt)

            # record values
            (cld(t, Δt) % record_rate == 0) && record!(recording, t, learners)

            @logprogress (t - tcurrent) / Tpre
        end
    end
    tcurrent += Tpre

    ## TRAINING

    @info "Starting training..."
    run!(data, net, learners, net_state, learner_states;
         nepochs = nepochs) do _, x, y, net, learners, net_state, learner_states
        x, y = target(x), target(y)
        for t in tcurrent:Δt:(tcurrent + Δtsample)
            # evaluate model
            zs = net(net_state, x, t, Δt)

            # get weight update
            zpres = (x, Base.front(zs)...)
            update!(net, learners, learner_states, x, y, copy.(zpres), copy.(zs), t, Δt) do t
                η(t; ηi = η0, toffset = Tinit + Tpre, rate = 50f0)
            end

            # record values
            (cld(t, Δt) % record_rate == 0) && record!(recording, t, learners)
        end
        # update time
        tcurrent += Δtsample
    end

    ## POST-TRAIN

    @info "Starting testing..."
    Ŷs = similar(data[2])
    net = net |> cpu
    net_state = state(net)
    run!(data, net, net_state; shuffle = false, progress = "TEST") do i, x, _, net, net_state
        z = mean(last(net(net_state, x, t, Δt)) for t in 0:Δt:Δtsample)
        Ŷs[:, i] .= z
    end

    @info "Starting decoding..."
    decoder = Dense(Dout, Dout)
    opt = Momentum()
    loss(y, ŷ) = Flux.Losses.logitbinarycrossentropy(decoder(ŷ), y)
    for _ in 1:1000
        Flux.train!(loss, Flux.params(decoder), [(data[2], Ŷs)], opt)
    end

    acc = mean(round.(Flux.sigmoid.(decoder(Ŷs))) .== data[2])
    @info "Accuracy = $(acc * 100)%"

    return recording
end

# function mnist_test(layer_config, target;
#                     τff = 5f-3, # LIF time constant for FF network
#                     τr = 50f-3, # LIF time constant for reservoirs
#                     λ = 1.7, # chaotic level
#                     γ = 2f0, # HSIC balance parameter
#                     η0 = 1f-3, # initial hsic learning rate
#                     ηr = 1f-4, # initial reservoir learning rate
#                     Nhidden = 1000, # size of reservoir
#                     noise = 5f-1,
#                     τavg = 10f-3, # signal smoothing constant
#                     Δt = 1f-3, # simulation time step
#                     Δtsample = 50f-3, # time to present each data sample
#                     bs = 15, # effective batch size
#                     nepochs = 100, # number of epochs
#                     Tinit = 50f0) # warmup time

#     η(t; ηi, toffset = zero(t), rate = 25f0) =
#         (t > toffset) ? ηi / (1 + (t - toffset) / rate) : zero(Float32)
#     H = I - fill(1 / bs, bs, bs) |> target # centering matrix
#     hsic_loss(K1, K2) = tr(K1 * H * K2 * H) / (bs - 1)^2

#     ## DATA SETUP

#     xtrain, ytrain = MNIST.traindata(Float32)
#     xtest, ytest = MNIST.testdata(Float32)
#     Nsamples = 500
#     Ttrain = Nsamples * Δtsample
#     idx = rand(1:nobs(xtrain), Nsamples)

#     xtrain = Flux.flatten(xtrain)[:, idx]
#     xtest = Flux.flatten(xtest)
#     ytrain = Flux.onehotbatch(ytrain, 0:9)[:, idx]
#     ytest = Flux.onehotbatch(ytest, 0:9)

#     xencoder = RateEncoder(xtrain, Δtsample) |> target
#     yencoder = RateEncoder(convert(Array{Float32}, ytrain), Δtsample) |> target

#     ## NETWORK SETUP

#     net = LIFChain([LIFDense{Float32}(in, out; τ = τff) for (in, out) in layer_config]) |> target
#     net_state = state(net)

#     learners = []
#     for (layer, layer_state) in zip(net, net_state)
#         # reservoir = Reservoir{Float32}(nout(xencoder) + nout(layer) + nout(yencoder) => nout(layer),
#         #                                Nhidden; τ = τr, λ = λ, noiseout = noise) |> target
#         # reservoir_learner = RMHebb(reservoir;
#         #                            η = t -> η(t; ηi = ηr, toffset = Tinit), τ = τavg) |> target

#         learner = HSICApprox(xencoder,
#                              yencoder,
#                              layer,
#                              layer_state,
#                             #  reservoir,
#                             #  reservoir_learner,
#                              bs; γ = γ, nbuffer = Int(cld(Δtsample, Δt)))

#         push!(learners, learner)
#     end
#     learner_states = state.(learners)

#     ## RECORDING SETUP

#     recording = (t = Float32[],
#                  lossxs = [Float32[] for _ in net],
#                  lossys = [Float32[] for _ in net])
#     function record!(recording, t, learners)
#         push!(recording.t, t)
#         for (i, learner) in enumerate(learners)
#             lossx = hsic_loss(learner.Kx, learner.Kz)
#             lossy = hsic_loss(learner.Ky, learner.Kz)
#             push!(recording.lossxs[i], lossx)
#             push!(recording.lossys[i], lossy)
#         end
#     end
#     record_rate = Int(cld(Δtsample, Δt))

#     ## WARMUP

#     @info "Starting warmup..."
#     tcurrent = 0f0
#     @progress "INIT" for t in trange(tcurrent, Δt, Tinit)
#         # evaluate model
#         x = xencoder(t)
#         y = yencoder(t)
#         zs = net(net_state, x, t, Δt)

#         # get weight update
#         zpres = (x, Base.front(zs)...)
#         update!(t -> 0, net, learners, learner_states, x, y, copy.(zpres), copy.(zs), t, Δt)

#         # record values
#         if cld(t, Δt) % record_rate == 0
#             record!(recording, t, learners)
#         end
#     end
#     tcurrent += Tinit

#     ## PRETRAINING

#     # @info "Starting pre-training..."
#     # @progress "PRETRAIN" for t in trange(tcurrent, Δt, Tpre)
#     #     # evaluate model
#     #     x = xencoder(t)
#     #     y = yencoder(t)
#     #     zs = net(net_state, x, t, Δt)

#     #     # get weight update
#     #     zpres = (x, Base.front(zs)...)
#     #     update!(t -> 0, net, learners, learner_states, x, y, copy.(zpres), copy.(zs), t, Δt)

#     #     # record values
#     #     if cld(t, Δt) % record_rate == 0
#     #         record!(recording, t, learners)
#     #     end
#     # end
#     # tcurrent += Tpre

#     ## TRAINING

#     @info "Starting training..."
#     @progress "TRAIN" for _ in 1:nepochs
#         # shuffle data
#         idx = shuffle(1:Nsamples)
#         xencoder.data .= xencoder.data[:, idx]
#         yencoder.data .= yencoder.data[:, idx]

#         current_loss = zeros(Float32, length(net))
#         current_n = 0
#         for t in trange(tcurrent, Δt, Ttrain)
#             # evaluate model
#             x = xencoder(t)
#             y = yencoder(t)
#             zs = net(net_state, x, t, Δt)

#             # get weight update
#             zpres = (x, Base.front(zs)...)
#             update!(net, learners, learner_states, x, y, copy.(zpres), copy.(zs), t, Δt) do t
#                 η(t; ηi = η0, toffset = Tinit, rate = 10 * Ttrain)
#             end

#             # record values
#             if cld(t, Δt) % record_rate == 0
#                 record!(recording, t, learners)
#                 @. current_loss += last(recording.lossxs) - γ * last(recording.lossys)
#                 current_n += 1
#             end
#         end
#         @show current_loss ./ current_n
#         tcurrent += Ttrain
#     end

#     ## POST-TRAIN

#     @info "Starting testing..."
#     Ŷs = similar(ytest, Float32)
#     net = net |> cpu
#     net_state = state(net)
#     @progress "TEST" for (i, x) in enumerate(eachcol(xtest))
#         for t in trange(0, Δt, Δtsample)
#             zs = net(net_state, x, t, Δt)
#             Ŷs[:, i] .= zs[end]
#         end
#     end

#     @info "Starting decoding..."
#     decoder = Dense(size(Ŷs, 1), size(ytest, 1))
#     opt = Momentum()
#     loss(y, ŷ) = Flux.Losses.logitcrossentropy(decoder(ŷ), y)
#     for _ in 1:1000
#         Flux.train!(loss, Flux.params(decoder), eachbatch((ytest, Ŷs), 32), opt)
#     end

#     acc = mean(mean(Flux.onecold(Flux.softmax(decoder(ŷ))) .== Flux.onecold(y))
#                for (ŷ, y) in zip(eachbatch(Ŷs, 32), eachbatch(ytest, 32)))
#     @info "Accuracy = $(acc * 100)%"

#     return recording
# end
