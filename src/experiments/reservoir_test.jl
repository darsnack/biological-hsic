function prepare_reservoir_data(X::T, Y::T, Z::T; bs, γ) where {S, T<:AbstractMatrix{S}}
    Nout = size(Z, 1)

    # precompute kernel matrices
    σx = estσ(X)
    σy = estσ(Y)
    σz = estσ(Z)
    Kx = CircularArray(k_hsic(X; σ = σx))
    Ky = CircularArray(k_hsic(Y; σ = σy))
    Kz = CircularArray(k_hsic(Z; σ = σz))
    Zs = CircularArray(Z)

    input_signal = vcat(X, Y, Z)
    irange(i) = i:-1:i - bs + 1
    global_error = GlobalError{S}(bs, Nout, γ)
    error_signal = [global_error(Kx[irange(i), i],
                                 Ky[irange(i), i],
                                 Kz[irange(i), i],
                                 Zs[:, irange(i)])
                    for i in 1:numobs(input_signal)]

    return eachobs((input_signal, error_signal))
end

function reservoir_test((X, Y, Z), Nhidden, target;
                        η0 = 1f-4,
                        γ = 2, # HSIC balance parameter
                        τ = 50f-3, # LIF time constant
                        λ = 1.7, # chaotic level
                        τavg = 5f-3, # signal smoothing constant
                        train_epochs = 100, # training time
                        test_epochs = 10, # testing time
                        Δt = 1f-3, # simulation time step
                        Δtsample = 50f-3, # time to present each data sample
                        bs = 6, # effective batch size
                        logger = nothing)
    # update logger configuration
    if logger isa WandbBackend
        config = Dict("inital_lr" => η0,
                      "gamma" => γ,
                      "tau_lif" => τ,
                      "lambda" => λ,
                      "tau_avg" => τavg,
                      "delta_t" => Δt,
                      "delta_sample" => Δtsample,
                      "batch_size" => bs,
                      "training_epochs" => train_epochs,
                      "test_epochs" => test_epochs)
        update_config!(logger, config)
    end

    # prepare reservoir input/output data
    data = prepare_reservoir_data(X, Y, Z; bs = bs, γ = γ)

    # build reservoir
    Nin = size(X, 1) + size(Y, 1) + size(Z, 1)
    Nout = size(Z, 1)
    u0, r0, z0 = zeros(Float32, Nhidden), zeros(Float32, Nhidden), zeros(Float32, Nout)
    reservoir = Reservoir(Float32, Nin => Nout, Nhidden;
                          τ = τ, λ = λ, Δt = Δt, init_state = (u0, r0, z0))

    # build learner
    opt = Descent(η0)
    tracecb = Traces((predicted = learner -> cpu(learner.step.ŷs)[1],
                      trueoutput = learner -> cpu(learner.step.ys)[1]))
    cbs = [Recorder(),
           ProgressPrinter(),
           ToDevice(target, target),
           tracecb]
    !isnothing(logger) && push!(cbs, LogTraces(logger))
    learner = Learner(reservoir, Flux.Losses.mse;
                      optimizer = opt,
                      callbacks = cbs,
                      usedefaultcallbacks = false)

    # fit model
    rmhebb = RMHebb(reservoir; τ = τavg) |> target
    training_scheme = RateEncoded(RMHebbTraining(rmhebb), Δt, Δtsample)
    for epoch in 1:train_epochs
        @info "Starting train epoch $epoch ..."
        epoch!(learner, training_scheme, data)
    end

    # test model
    # for epoch in 1:test_epochs
    #     @info "Starting test epoch $epoch ..."
        
end