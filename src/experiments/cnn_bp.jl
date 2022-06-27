include("../setup.jl")

function mnist_cnn_bp_test(channel_config, param_config, dense_config, target;
                           percent_samples = 100, classes = 0:9, kwargs...)
    xtrain, ytrain = MNIST.traindata(Float32)
    xtest, ytest = MNIST.testdata(Float32)

    xtrain, ytrain = subsample((xtrain, ytrain), percent_samples / 100)

    xtrain, ytrain = filter_classes(xtrain, ytrain, classes)
    xtest, ytest = filter_classes(xtest, ytest, classes)

    xtrain = Flux.unsqueeze(xtrain, 3)
    xtest = Flux.unsqueeze(xtest, 3)
    ytrain = convert(Array{Float32}, Flux.onehotbatch(ytrain, classes))::Matrix{Float32}
    ytest = convert(Array{Float32}, Flux.onehotbatch(ytest, classes))::Matrix{Float32}

    data = (train = (xtrain, ytrain), test = (xtest, ytest))

    return cnn_bp_test(data, channel_config, param_config, dense_config, target; kwargs...)
end

function cifar10_cnn_bp_test(channel_config, param_config, dense_config, target;
                             percent_samples = 100, classes = 0:9, kwargs...)
    xtrain, ytrain = CIFAR10.traindata(Float32)
    xtest, ytest = CIFAR10.testdata(Float32)

    xtrain, ytrain = subsample((xtrain, ytrain), percent_samples / 100)

    xtrain, ytrain = filter_classes(xtrain, ytrain, classes)
    xtest, ytest = filter_classes(xtest, ytest, classes)

    ytrain = convert(Array{Float32}, Flux.onehotbatch(ytrain, classes))::Matrix{Float32}
    ytest = convert(Array{Float32}, Flux.onehotbatch(ytest, classes))::Matrix{Float32}

    data = (train = (xtrain, ytrain), test = (xtest, ytest))

    return cnn_bp_test(data, channel_config, param_config, dense_config, target; kwargs...)
end

function cnn_bp_test(data, channel_config, param_config, dense_config, target;
                     opt = Momentum(), # optimizers
                     schedule = Sequence(opt.eta => nepochs),
                     bs = 32, # effective batch size
                     nepochs = 100, # number of epochs
                     validation_points = (nepochs,),
                     progressrate = 1,
                     logger = global_logger())

    ## DATA SETUP

    xtrain, ytrain = data.train
    xtest, ytest = data.test

    insize = size(xtrain)[1:(end - 1)]
    nsamples = size(xtrain)[end]

    ## NETWORK SETUP

    convs = conv_chain((3, 3), channel_config, param_config)
    _dense_cfg = (prod(Flux.outputsize(convs, (insize..., 1))), dense_config...)
    denses = dense_chain(_dense_cfg[1:(end - 1)])
    final = Dense(filter(x -> x isa Integer, _dense_cfg)[(end - 1):end]...)
    net = Chain(convs..., Flux.flatten, denses..., final) |> target

    ## LOSS SETUP

    loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

    ## RECORDING SETUP

    recording = (t = Float32[],
                 accuracies = Union{Float32, Missing}[],
                 loss = Float32[])
    function record!(recording, t, l)
        push!(recording.t, t)
        push!(recording.loss, l)
    end

    ## VALIDATION SETUP

    accuracy(m, xs, ys) =
        mean(mean(Flux.onecold(Flux.softmax(m(target(x)))) .== Flux.onecold(target(y)))
               for (x, y) in zip(eachbatch(xs, bs), eachbatch(ys, bs)))

    ## SCHEDULING SETUP

    scheduler = Scheduler(Interpolator(schedule, floor(Int, nsamples / bs)), opt)

    ## TRAINING

    @info "Starting training..."
    current_loss = 0f0
    epoch = 0
    run!((xtrain, ytrain), net, scheduler;
         bs = bs, nepochs = nepochs, progress = ("TRAIN", progressrate)) do i, x, y, net, scheduler
        x, y = target(x), target(y)
        ps = Flux.params(net)
        gs = gradient(ps) do
            l = loss(net, x, y)
            Zygote.ignore() do
                if i == 1
                    # current_loss = l
                    # current_n = 1
                    epoch += 1
                # else
                #     current_loss += l
                #     current_n += 1
                end
                current_loss = l
                record!(recording, i, l)
            end
            return l
        end

        Flux.Optimise.update!(scheduler, ps, gs)

        # validate
        if (i == 1) && (epoch ∈ validation_points)
            acc = accuracy(net, xtest, ytest)
            with_logger(logger) do
                @info "test" acc=acc log_step_increment=0
            end
            push!(recording.accuracies, acc)
        else
            push!(recording.accuracies, missing)
        end

        # display current loss
        with_logger(logger) do
            @info "train" loss=current_loss lr=opt.eta
        end
    end

    ## POST-TRAIN

    acc = accuracy(net, xtest, ytest)
    recording.accuracies[end] = acc
    @info "Accuracy = $(acc * 100)%"

    return recording,
           (model = net,
            data = data)
end
