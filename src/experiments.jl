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

include("../src/utils.jl")
include("../src/hsic.jl")
include("../src/lpf.jl")
include("../src/reservoir.jl")
include("../src/networks.jl")
include("../src/data.jl")
include("../src/learning.jl")

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
                  Tpre = 500f0, # pretraining time
                  Ttrain = Nsamples * Δtsample) # training time

    η(t; ηi, toffset = zero(t), rate = 25f0) =
        (t > toffset) ? ηi / (1 + (t - toffset) / rate) : zero(Float32)
    H = I - fill(1 / bs, bs, bs) |> target # centering matrix
    hsic_loss(Kx, Kz) = tr(Kx * H * Kz * H) / (bs - 1)^2

    ## NETWORK SETUP

    xencoder = RateEncoder(data[1], Δtsample) |> target
    yencoder = RateEncoder(data[2], Δtsample) |> target

    net = LIFDense{Float32}(Din, Dout; τ = τff) |> target
    net_state = state(net)

    reservoir = Reservoir{Float32}(nout(xencoder) + nout(net) + nout(yencoder) => nout(net),
                                   Nhidden; τ = τr, λ = λ, noiseout = noise) |> target
    reservoir_learner = RMHebb(reservoir; η = t -> η(t; ηi = ηr, toffset = Tinit), τ = τavg) |> target

    learner = HSIC(xencoder, yencoder, net, net_state, reservoir, reservoir_learner, bs;
                   γ = γ, nbuffer = Int(cld(Δtsample, Δt)))
    learner_state = state(learner)

    ## RECORDING SETUP

    lossx_lpf = LowPassFilter(Δtsample, target(zeros(Float32, 1, 1)))
    lossy_lpf = LowPassFilter(Δtsample, target(zeros(Float32, 1, 1)))
    recording = (t = Float32[], lossx = Float32[], lossy = Float32[], w1 = Float32[], w2 = Float32[])
    record_rate = Int(cld(Δtsample, Δt))

    ## WARMUP

    @info "Starting warmup..."
    tcurrent = 0f0
    @progress "INIT" for t in trange(tcurrent, Δt, Tinit)
        # evaluate model
        x = xencoder(t)
        y = yencoder(t)
        z = net(net_state, x, t, Δt)

        # get weight update
        learner(learner_state, x, y, x, copy(z), t, Δt)

        # record values
        lossx = lossx_lpf(hsic_loss(learner.Kx, learner.Kz), Δt)
        lossy = lossy_lpf(hsic_loss(learner.Ky, learner.Kz), Δt)
        if cld(t, Δt) % record_rate == 0
            push!(recording.t, t)
            push!(recording.lossx, only(cpu(lossx)))
            push!(recording.lossy, only(cpu(lossy)))
            push!(recording.w1, cpu(net.W)[1])
            push!(recording.w2, cpu(net.W)[2])
        end
    end
    tcurrent += Tinit

    ## PRETRAINING

    @info "Starting pre-training..."
    @progress "PRETRAIN" for t in trange(tcurrent, Δt, Tpre)
        # evaluate model
        x = xencoder(t)
        y = yencoder(t)
        z = net(net_state, x, t, Δt)

        # get weight update
        learner(learner_state, x, y, x, copy(z), t, Δt)

        # record values
        lossx = lossx_lpf(hsic_loss(learner.Kx, learner.Kz), Δt)
        lossy = lossy_lpf(hsic_loss(learner.Ky, learner.Kz), Δt)
        if cld(t, Δt) % record_rate == 0
            push!(recording.t, t)
            push!(recording.lossx, only(cpu(lossx)))
            push!(recording.lossy, only(cpu(lossy)))
            push!(recording.w1, cpu(net.W)[1])
            push!(recording.w2, cpu(net.W)[2])
        end
    end
    tcurrent += Tpre

    ## TRAINING

    @info "Starting training..."
    @progress "TRAIN" for _ in 1:nepochs
        # shuffle data
        idx = shuffle(1:Nsamples)
        xencoder.data .= xencoder.data[:, idx]
        yencoder.data .= yencoder.data[:, idx]
        for t in trange(tcurrent, Δt, Ttrain)
            # evaluate model
            x = xencoder(t)
            y = yencoder(t)
            z = net(net_state, x, t, Δt)

            # get weight update
            update!(net, learner, learner_state, x, y, x, copy(z), t, Δt) do t
                η(t; ηi = η0, toffset = Tinit + Tpre, rate = 50f0)
            end

            # record values
            lossx = lossx_lpf(hsic_loss(learner.Kx, learner.Kz), Δt)
            lossy = lossy_lpf(hsic_loss(learner.Ky, learner.Kz), Δt)
            if cld(t, Δt) % record_rate == 0
                push!(recording.t, t)
                push!(recording.lossx, only(cpu(lossx)))
                push!(recording.lossy, only(cpu(lossy)))
                push!(recording.w1, cpu(net.W)[1])
                push!(recording.w2, cpu(net.W)[2])
            end
        end
        tcurrent += Ttrain
    end

    ## POST-TRAIN

    @info "Starting testing..."
    Ŷs = similar(data[2])
    net = net |> cpu
    net_state = state(net)
    @progress "TEST" for (i, (x, _)) in enumerate(zip(eachcol(data[1]), eachcol(data[2])))
        for t in trange(0, Δt, Δtsample)
            z = net(net_state, x, t, Δt)
            Ŷs[:, i] .= z
        end
    end

    @info "Starting decoding..."
    decoder = Dense(Dout, Dout)
    opt = Momentum(1e-2)
    loss(y, ŷ) = Flux.Losses.logitbinarycrossentropy(y, decoder(ŷ))
    for _ in 1:1000
        Flux.train!(loss, Flux.params(decoder), [(data[2], Ŷs)], opt)
    end

    acc = mean(round.(Flux.sigmoid.(decoder(Ŷs))) .== data[2])
    @info "Accuracy = $(acc * 100)%"
    @show mean((sign.(Ŷs) .+ 1) ./ 2 .== data[2])

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
                    Nsamples = 100, # number of data samples
                    Δtsample = 50f-3, # time to present each data sample
                    bs = 15, # effective batch size
                    nepochs = 15, # number of epochs
                    Tinit = 50f0, # warmup time
                    Tpre = 500f0, # pretraining time
                    Ttrain = Nsamples * Δtsample) # training time

    η(t; ηi, toffset = zero(t), rate = 25f0) =
        (t > toffset) ? ηi / (1 + (t - toffset) / rate) : zero(Float32)
    H = I - fill(1 / bs, bs, bs) |> target # centering matrix
    hsic_loss(Kx, Kz) = tr(Kx * H * Kz * H) / (bs - 1)^2

    ## NETWORK SETUP

    xencoder = RateEncoder(data[1], Δtsample) |> target
    yencoder = RateEncoder(data[2], Δtsample) |> target

    net = LIFChain([LIFDense{Float32}(in, out; τ = τff) for (in, out) in layer_config]) |> target
    net_state = state(net)

    learners = []
    for (layer, layer_state) in zip(net, net_state)
        reservoir = Reservoir{Float32}(nout(xencoder) + nout(layer) + nout(yencoder) => nout(layer),
                                       Nhidden; τ = τr, λ = λ, noiseout = noise) |> target
        reservoir_learner = RMHebb(reservoir;
                                   η = t -> η(t; ηi = ηr, toffset = Tinit), τ = τavg) |> target

        learner = HSIC(xencoder,
                       yencoder,
                       layer,
                       layer_state,
                       reservoir,
                       reservoir_learner,
                       bs; γ = γ, nbuffer = Int(cld(Δtsample, Δt)))

        push!(learners, learner)
    end
    learner_states = state.(learners)

    ## RECORDING SETUP

    lossx_lpf = [LowPassFilter(Δtsample, target(zeros(Float32, 1, 1))) for _ in net]
    lossy_lpf = [LowPassFilter(Δtsample, target(zeros(Float32, 1, 1))) for _ in net]
    recording = (t = Float32[],
                 lossxs = [Float32[] for _ in net],
                 lossys = [Float32[] for _ in net])
    function record!(recording, t, learners)
        push!(recording.t, t)
        for (i, learner) in enumerate(learners)
            lossx = lossx_lpf[i](hsic_loss(learner.Kx, learner.Kz), Δt)
            lossy = lossy_lpf[i](hsic_loss(learner.Ky, learner.Kz), Δt)
            push!(recording.lossxs[i], only(cpu(lossx)))
            push!(recording.lossys[i], only(cpu(lossy)))
        end
    end
    record_rate = Int(cld(Δtsample, Δt))

    ## WARMUP

    @info "Starting warmup..."
    tcurrent = 0f0
    @progress "INIT" for t in trange(tcurrent, Δt, Tinit)
        # evaluate model
        x = xencoder(t)
        y = yencoder(t)
        zs = net(net_state, x, t, Δt)

        # get weight update
        zpres = (x, Base.front(zs)...)
        update!(t -> 0, net, learners, learner_states, x, y, copy.(zpres), copy.(zs), t, Δt)

        # record values
        if cld(t, Δt) % record_rate == 0
            record!(recording, t, learners)
        end
    end
    tcurrent += Tinit

    ## PRETRAINING

    @info "Starting pre-training..."
    @progress "PRETRAIN" for t in trange(tcurrent, Δt, Tpre)
        # evaluate model
        x = xencoder(t)
        y = yencoder(t)
        zs = net(net_state, x, t, Δt)

        # get weight update
        zpres = (x, Base.front(zs)...)
        update!(t -> 0, net, learners, learner_states, x, y, copy.(zpres), copy.(zs), t, Δt)

        # record values
        if cld(t, Δt) % record_rate == 0
            record!(recording, t, learners)
        end
    end
    tcurrent += Tpre

    ## TRAINING

    @info "Starting training..."
    @progress "TRAIN" for _ in 1:nepochs
        # shuffle data
        idx = shuffle(1:Nsamples)
        xencoder.data .= xencoder.data[:, idx]
        yencoder.data .= yencoder.data[:, idx]
        for t in trange(tcurrent, Δt, Ttrain)
            # evaluate model
            x = xencoder(t)
            y = yencoder(t)
            zs = net(net_state, x, t, Δt)

            # get weight update
            zpres = (x, Base.front(zs)...)
            update!(net, learners, learner_states, x, y, copy.(zpres), copy.(zs), t, Δt) do t
                η(t; ηi = η0, toffset = Tinit + Tpre, rate = 50f0)
            end

            # record values
            if cld(t, Δt) % record_rate == 0
                record!(recording, t, learners)
            end
        end
        tcurrent += Ttrain
    end

    ## POST-TRAIN

    @info "Starting testing..."
    Ŷs = similar(data[2])
    net = net |> cpu
    net_state = state(net)
    @progress "TEST" for (i, (x, _)) in enumerate(zip(eachcol(data[1]), eachcol(data[2])))
        for t in trange(0, Δt, Δtsample)
            zs = net(net_state, x, t, Δt)
            Ŷs[:, i] .= zs[end]
        end
    end

    @info "Starting decoding..."
    decoder = Dense(Dout, Dout)
    opt = Momentum(1e-2)
    loss(y, ŷ) = Flux.Losses.logitbinarycrossentropy(y, decoder(ŷ))
    for _ in 1:1000
        Flux.train!(loss, Flux.params(decoder), [(data[2], Ŷs)], opt)
    end

    acc = mean(round.(Flux.sigmoid.(decoder(Ŷs))) .== data[2])
    @info "Accuracy = $(acc * 100)%"
    @show mean((sign.(Ŷs) .+ 1) ./ 2 .== data[2])

    return recording
end
