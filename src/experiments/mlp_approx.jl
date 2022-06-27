include("../setup.jl")

function mnist_mlp_approx_test(layer_config, target; percent_samples = 100, classes = 0:9, kwargs...)
    xtrain, ytrain = MNIST.traindata(Float32)
    xtest, ytest = MNIST.testdata(Float32)

    xtrain, ytrain = subsample((xtrain, ytrain), percent_samples / 100)

    xtrain, ytrain = filter_classes(xtrain, ytrain, classes)
    xtest, ytest = filter_classes(xtest, ytest, classes)

    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)
    ytrain = convert(Array{Float32}, Flux.onehotbatch(ytrain, classes))::Matrix{Float32}
    ytest = convert(Array{Float32}, Flux.onehotbatch(ytest, classes))::Matrix{Float32}

    data = (train = (xtrain, ytrain), test = (xtest, ytest))

    return mlp_approx_test(data, layer_config, target; kwargs...)
end

function cifar10_mlp_approx_test(layer_config, target; percent_samples = 100, classes = 0:9, kwargs...)
    xtrain, ytrain = CIFAR10.traindata(Float32)
    xtest, ytest = CIFAR10.testdata(Float32)

    xtrain, ytrain = subsample((xtrain, ytrain), percent_samples / 100)

    xtrain, ytrain = filter_classes(xtrain, ytrain, classes)
    xtest, ytest = filter_classes(xtest, ytest, classes)

    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)
    ytrain = convert(Array{Float32}, Flux.onehotbatch(ytrain, classes))::Matrix{Float32}
    ytest = convert(Array{Float32}, Flux.onehotbatch(ytest, classes))::Matrix{Float32}

    data = (train = (xtrain, ytrain), test = (xtest, ytest))

    return mlp_approx_test(data, layer_config, target; kwargs...)
end

function mlp_approx_test(data, layer_config, target;
                         τff = 5f-3, # LIF time constant for FF network
                         γs = fill(2f0, length(layer_config)), # HSIC balance parameter
                         σx = 0.2,
                         σy = 0.5,
                         σzs = fill(0.5, length(layer_config)),
                         scale_σ = true,
                         opts = fill(Descent(1f-2), length(layer_config)), # optimizers
                         schedules = map(o -> Sequence(o.eta => nepochs), opts),
                         Δt = 1f-3, # simulation time step
                         Δtsample = 50f-3, # time to present each data sample
                         bs = 15, # effective batch size
                         nepochs = 100, # number of epochs
                         validation_points = (nepochs,),
                         nthreads = 1,
                         progressrate = 1,
                         logger = global_logger())

    ## DATA SETUP

    xtrain, ytrain = data.train
    xtest, ytest = data.test

    Din = size(xtrain, 1)
    Dout = size(ytrain, 1)
    nsamples = size(xtrain, 2)

    ## CONSTANTS

    H = I - fill(1 / bs, bs, bs) |> target # centering matrix

    ## SCHEDULING SETUP

    schedulers = map(opts, schedules) do o, s
        Scheduler(Interpolator(Sequence(0f0 => bs,
                                        Interpolator(s, nsamples) => nepochs * nsamples),
                               Δtsample / Δt), o)
    end

    ## NETWORK SETUP

    sample_length = ceil(Int, Δtsample / Δt)
    _dense_cfg = (Din, layer_config...)
    denses = lifdense_chain(Float32, _dense_cfg; τ = τff)
    if scale_σ
        σx *= sqrt(Din)
        σy *= sqrt(Dout)
        σzs = map((d, σz) -> σz * sqrt(d), layer_config, σzs)
    end
    net = LIFChain(map(denses, γs, σzs, schedulers) do layer, γ, σz, opt
        HSICLayer(layer,
                  HSICApprox(Float32, Din, Dout, outsize(layer)..., bs;
                             γ = γ, nbuffer = sample_length, sigmas = (σx, σy, σz)),
                  opt)
    end) |> target
    net_state = state(net, (Din,))

    ## RECORDING SETUP

    recording = (t = Float32[],
                 accuracies = Union{Float32, Missing}[],
                 lossxs = [Float32[] for _ in net],
                 lossys = [Float32[] for _ in net])
    function record!(recording, t, layers)
        push!(recording.t, t)
        for (i, layer) in enumerate(layers)
            Kx, Ky, Kz = layer.learner.kernels.matrices
            # @show i, mean(learner.Kx), median(learner.Kx), minimum(learner.Kx)
            lossx = hsic(Kx, Kz, H)
            lossy = hsic(Ky, Kz, H)
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
        ŷs = predict(net, xs, ys; Δt = Δt, Δtsample = Δtsample, progress = ("TEST", progressrate))
        fitdecoder!(decoder, ŷs, ys)

        return accuracy(decoder, ys, ŷs)
    end

    ## TRAINING

    @info "Starting training..."
    tcurrent = 0
    current_loss = zeros(Float32, length(net))
    epoch = 0
    let xtest = xtest, ytest = ytest
        run!((xtrain, ytrain), net, net_state;
             nepochs = nepochs, progress = ("TRAIN", progressrate)) do i, x, y, net, net_state
            x, y = target(x), target(y)
            for t in tcurrent:Δt:(tcurrent + Δtsample)
                # evaluate model
                zs = net(net_state, x, t, Δt)

                # get weight update
                update!(net, net_state, x, y, zs[1:(end - 1)], zs[2:end], t, Δt;
                        nthreads = nthreads)
            end
            # record values
            record!(recording, tcurrent, net)

            # update time
            tcurrent += Δtsample
            if i == 1
                epoch += 1
            end

            # accumulate loss
            @. current_loss = last(recording.lossxs) - γs * last(recording.lossys)

            # validate
            if (i == 1) && (epoch ∈ validation_points)
                acc = validate!(decoder, net, xtest, ytest)
                with_logger(logger) do
                    @info "test" acc=acc log_step_increment=0
                end
                push!(recording.accuracies, acc)
            else
                push!(recording.accuracies, missing)
            end

            # display current loss
            with_logger(logger) do
                for j in 1:length(net)
                    incr = (j == length(net)) ? 1 : 0
                    l = current_loss[j]
                    c = last(recording.lossxs[j])
                    o = opts[j].eta
                    @info "train-layer-$j" loss=l lr=o compression=c log_step_increment=incr
                end
            end
        end
    end

    ## POST-TRAIN

    acc = validate!(decoder, net, xtest, ytest)
    recording.accuracies[end] = acc
    @info "Accuracy = $(acc * 100)%"

    return recording,
           (model = (net, decoder),
            data = (data...,
                    predictions = predict(net, xtest, ytest;
                                          Δt = Δt,
                                          Δtsample = Δtsample,
                                          progress = ("TEST", progressrate))))
end
