
function reservoir_test(X, Y, Z, target;
                        η0 = 1f-4,
                        τ = 50f-3, # LIF time constant
                        λ = 1.7, # chaotic level
                        τavg = 5f-3, # signal smoothing constant
                        Tinit = 50f0, # warmup time
                        Ttrain = 500f0, # training time
                        Ttest = 100f0, # testing time
                        Δt = 1f-3, # simulation time step
                        Nsamples = 100, # number of data samples
                        Nhidden = 2000, # number of hidden neurons in reservoir
                        Δtsample = 50f-3, # time to present each data sample
                        bs = 6) # effective batch size
    # learning rate
    η = ηdecay(η0; toffset = Tinit, rate = 20f0)

    # network sizes
    Nx = size(X, 1)
    Ny = size(Y, 1)
    Nz = size(Z, 1)
    Nin = Nx + Ny + Nz # needs to be >= 1 even if no input
    Nout = Nz

    # data

    σx = estσ(X)
    σy = estσ(Y)
    σz = estσ(Z)
    Kx = k_hsic(X, X; σ = σx)
    Ky = k_hsic(Y, Y; σ = σy)
    Kz = k_hsic(Z, Z; σ = σz)

    # input signal
    timetoidx(t) = (t < 0) ? 1 : (Int(round(t / Δtsample)) % Nsamples) + 1
    function input(t)
        (t < 0) && return zeros(Float32, Nin) |> target
        i = timetoidx(t)

        return concatenate(X[:, i], Y[:, i], Z[:, i])
    end

    # true signal
    ξ = GlobalError{Float32}(bs, Nz) |> target
    function f(t)#::CuVector{Float32}
        is = timetoidx.([t - i * Δtsample for i in 0:(bs - 1)])
        kx = Kx[is, is]
        ky = Ky[is, is]
        kz = Kz[is, is]
        z = Z[:, is]

        return ξ(kx, ky, kz, z; σz = σz)
    end

    ## PROBLEM SETUP

    reservoir = Reservoir{Float32}(Nin => Nout, Nhidden;
                                   λ = λ, τ = τ, noisehidden = 5f-6, noiseout = 1f-2) |> target
    learner = RMHebb(reservoir; η = η, τ = τavg) |> target

    ## RECORDING SETUP

    recording = (t = Float32[],
                 z = Vector{Float32}[],
                 zlpf = Vector{Float32}[],
                 wnorm = Float32[],
                 f = Vector{Float32}[])

    ## STATE INITIALIZATION

    reservoir_state = state(reservoir)

    ## WARMUP

    @info "Starting warmup..."
    @progress "INIT" for t in 0:Δt:(Tinit - Δt)
        step!(reservoir, reservoir_state, input(t), t, Δt)
        push!(recording.t, t)
        push!(recording.z, cpu(reservoir_state.z))
        push!(recording.zlpf, cpu(learner.zlpf.f̄))
        push!(recording.wnorm, norm(reservoir.Wout))
        push!(recording.f, cpu(f(t)))
    end

    ## TRAIN

    @info "Starting training..."
    @progress "TRAIN" for t in Tinit:Δt:(Tinit + Ttrain - Δt)
        step!(reservoir, reservoir_state, learner, input(t), f(t), t, Δt)
        push!(recording.t, t)
        push!(recording.z, cpu(reservoir_state.z))
        push!(recording.zlpf, cpu(learner.zlpf.f̄))
        push!(recording.wnorm, norm(reservoir.Wout))
        push!(recording.f, cpu(f(t)))
    end

    ## TEST

    @info "Starting testing..."
    @progress "TEST" for t in (Tinit + Ttrain):Δt:(Tinit + Ttrain + Ttest)
        step!(reservoir, reservoir_state, input(t), t, Δt; explore = false)
        push!(recording.t, t)
        push!(recording.z, cpu(reservoir_state.z))
        push!(recording.zlpf, cpu(learner.zlpf.f̄))
        push!(recording.wnorm, norm(reservoir.Wout))
        push!(recording.f, cpu(f(t)))
    end

    return recording
end

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

    H = I - fill(1 / bs, bs, bs) |> target # centering matrix

    ## NETWORK SETUP

    Din = size(data[1], 1)
    Dout = size(data[2], 1)

    net = LIFDense{Float32}(Din, Dout; τ = τff) |> target
    net_state = state(net)

    η = ηdecay(η0; toffset = Tinit + Tpre, rate = 30f0)
    opt = Descent(η0)

    reservoir = Reservoir{Float32}(Din + nout(net) + Dout => nout(net), Nhidden;
                                   τ = τr, λ = λ, noiseout = noise) |> target
    reservoir_learner = RMHebb(reservoir; η = ηdecay(ηr; toffset = Tinit), τ = τavg) |> target

    learner = HSIC(data..., net, net_state, reservoir, reservoir_learner, bs;
                   γ = γ, nbuffer = Int(cld(Δtsample, Δt))) |> target
    learner_state = state(learner)

    ## RECORDING SETUP

    recording = (t = Float32[], lossx = Float32[], lossy = Float32[], w1 = Float32[], w2 = Float32[])
    function record!(recording, t, net, learner)
        lossx = hsic(learner.Kx, learner.Kz, H)
        lossy = hsic(learner.Ky, learner.Kz, H)
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
            learner(learner_state, copy(x), copy(y), copy(x), copy(z), t, Δt)

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
            learner(learner_state, copy(x), copy(y), copy(x), copy(z), t, Δt)

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
            opt.eta = η(t)
            update!(opt, net, learner, learner_state, copy(x), copy(y), copy(x), copy(z), t, Δt)

            # record values
            (cld(t, Δt) % record_rate == 0) && record!(recording, t, net, learner)
        end
        # update time
        tcurrent += Δtsample
    end

    ## POST-TRAIN

    @info "Starting testing..."
    Ŷs = predict(net, data...)

    @info "Starting decoding..."
    decoder = fitdecoder(Ŷs, data[2])

    acc = mean(round.(Flux.sigmoid.(decoder(Ŷs))) .== data[2])
    @info "Accuracy = $(acc * 100)%"

    return recording, acc
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

    H = I - fill(1 / bs, bs, bs) |> target # centering matrix

    ## NETWORK SETUP

    Din = size(data[1], 1)
    Dout = size(data[2], 1)

    net = LIFChain([LIFDense{Float32}(in, out; τ = τff) for (in, out) in layer_config]) |> target
    net_state = state(net)

    η = ηdecay(η0; toffset = Tinit + Tpre, rate = 50f0)
    opts = [Descent(η0) for _ in net]

    learners = []
    for (layer, layer_state) in zip(net, net_state)
        reservoir = Reservoir{Float32}(Din + nout(layer) + Dout => nout(layer),
                                       Nhidden; τ = τr, λ = λ, noiseout = noise) |> target
        reservoir_learner = RMHebb(reservoir;
                                   η = ηdecay(ηr; toffset = Tinit), τ = τavg) |> target

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
            lossx = hsic(learner.Kx, learner.Kz, H)
            lossy = hsic(learner.Ky, learner.Kz, H)
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
            update!(net, learners, learner_states,
                    copy(x), copy(y), copy.(zpres), copy.(zs),
                    t, Δt)

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
            update!(net, learners, learner_states,
                    copy(x), copy(y), copy.(zpres), copy.(zs),
                    t, Δt)

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
            foreach(opts) do o
                o.eta = η(t)
            end
            update!(opts, net, learners, learner_states, copy(x), copy(y), copy.(zpres), copy.(zs), t, Δt)

            # record values
            (cld(t, Δt) % record_rate == 0) && record!(recording, t, learners)
        end
        # update time
        tcurrent += Δtsample
    end

    ## POST-TRAIN

    @info "Starting testing..."
    Ŷs = predict(net, data...)

    @info "Starting decoding..."
    decoder = fitdecoder(Ŷs, data[2])

    acc = mean(round.(Flux.sigmoid.(decoder(Ŷs))) .== data[2])
    @info "Accuracy = $(acc * 100)%"

    return recording, acc
end

function phsic_test(layer_config, target;
                    τff = 5f-3, # LIF time constant for FF network
                    γs = fill(2f0, length(layer_config)), # HSIC balance parameter
                    opts = fill(Descent(1f-2), length(layer_config)), # optimizers
                    schedules = map(o -> Sequence(o.eta => nepochs), opts),
                    Δt = 1f-3, # simulation time step
                    Δtsample = 50f-3, # time to present each data sample
                    nepochs = 100, # number of epochs
                    validation_points = (nepochs,),
                    percent_samples = 100,
                    classes = 0:9,
                    nthreads = 1)
    bs = 2

    ## DATA SETUP

    xtrain, ytrain = MNIST.traindata(Float32)
    xtest, ytest = MNIST.testdata(Float32)

    (xtrain, ytrain), _ = stratifiedobs((xtrain, ytrain), percent_samples / 100)

    trainidx = findall(y -> any(y .== classes), ytrain)
    xtrain, ytrain = xtrain[:, :, trainidx], ytrain[trainidx]
    testidx = findall(y -> any(y .== classes), ytest)
    xtest, ytest = xtest[:, :, testidx], ytest[testidx]

    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)
    ytrain = convert(Array{Float32}, Flux.onehotbatch(ytrain, classes))
    ytest = convert(Array{Float32}, Flux.onehotbatch(ytest, classes))

    Din = size(xtrain, 1)
    Dout = size(ytrain, 1)
    nsamples = size(xtrain, 2)

    ## CONSTANTS

    H = I - fill(1 / bs, bs, bs) |> target # centering matrix

    ## NETWORK SETUP

    net = LIFChain([LIFDense{Float32}(in, out; τ = τff) for (in, out) in layer_config]) |> target
    net_state = state(net)

    learners = []
    for (layer, layer_state, γ) in zip(net, net_state, γs)
        learner = pHSIC(ytrain,
                        layer,
                        layer_state; γ = γ, nbuffer = Int(cld(Δtsample, Δt))) |> target

        push!(learners, learner)
    end
    learner_states = state.(learners)

    ## RECORDING SETUP

    recording = (t = Float32[],
                 accuracies = Union{Float32, Missing}[],
                 lossxs = [Float32[] for _ in net],
                 lossys = [Float32[] for _ in net])
    function record!(recording, t, learners)
        push!(recording.t, t)
        for (i, learner) in enumerate(learners)
            # @show i, mean(learner.Kz), median(learner.Kz), minimum(learner.Kz)
            lossx = hsic(learner.Kz, learner.Kz, H)
            lossy = hsic(learner.Ky, learner.Kz, H)
            push!(recording.lossxs[i], lossx)
            push!(recording.lossys[i], lossy)
        end
    end

    ## VALIDATION SETUP

    accuracy(decoder, ys, ŷs) =
        mean(mean(Flux.onecold(Flux.softmax(decoder(ŷ))) .== Flux.onecold(y))
               for (ŷ, y) in zip(eachbatch(ŷs, 32), eachbatch(ys, 32)))

    decoder = Dense(Dout, Dout)
    function validate!(decoder, net, xs, ys)
        ŷs = predict(net, xs, ys)
        fitdecoder!(decoder, ŷs, ys)

        return accuracy(decoder, ys, ŷs)
    end

    ## SCHEDULING SETUP

    schedulers = map(opts, schedules) do o, s
        Scheduler(Interpolator(Sequence(0f0 => bs,
                                        Interpolator(s, nsamples) => nepochs * nsamples),
                               Δtsample / Δt), o)
    end

    ## TRAINING

    @info "Starting training..."
    tcurrent = 0
    current_loss = zeros(Float32, length(net))
    current_n = 0
    epoch = 0
    run!((xtrain, ytrain), net, learners, net_state, learner_states;
         nepochs = nepochs, progress = "TRAIN") do i, x, y, net, learners, net_state, learner_states
        x, y = target(x), target(y)
        for t in tcurrent:Δt:(tcurrent + Δtsample)
            # evaluate model
            zs = net(net_state, x, t, Δt)

            # get weight update
            zpres = (x, Base.front(zs)...)
            update!(schedulers,
                    net, learners, learner_states,
                    copy(x), copy(y), copy.(zpres), copy.(zs),
                    t, Δt;
                    nthreads = nthreads)
        end
        # record values
        record!(recording, tcurrent, learners)

        # update time
        tcurrent += Δtsample

        # accumulate loss
        if i == 1
            @. current_loss = last(recording.lossxs) - γs * last(recording.lossys)
            current_n = 1
            epoch += 1
        else
            @. current_loss += last(recording.lossxs) - γs * last(recording.lossys)
            current_n += 1
        end

        # validate
        if (i == 1) && (epoch ∈ validation_points)
            push!(recording.accuracies, validate!(decoder, net, xtest, ytest))
        else
            push!(recording.accuracies, missing)
        end

        # display current loss
        if i % 100 == 0
            # @show mean(learners[1].Kx), median(learners[1].Kx), minimum(learners[1].Kx)
            # @show mean(learners[1].Kz), median(learners[1].Kz), minimum(learners[1].Kz)
            # @show getproperty.(opts, :eta)
            @show current_loss ./ current_n
        end
    end

    ## POST-TRAIN

    acc = validate!(decoder, net, xtest, ytest)
    recording.accuracies[end] = acc
    @info "Accuracy = $(acc * 100)%"

    return recording,
           (model = (net, learners, decoder),
            data = (train = (xtrain, ytrain),
                    test = (xtest, ytest),
                    predictions = predict(net, xtest, ytest)))
end
