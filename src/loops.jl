function run!(step!, data, args...; nepochs = 1, bs = 1, shuffle = true, progress = nothing)
    progressname = isnothing(progress) ? "" : progress
    @withprogress name=progressname begin
        for epoch in 1:nepochs
            @info "Starting epoch $epoch ..."

            dataloader = shuffle ? shuffleobs(data) : data
            dataloader = (bs > 1) ? eachbatch(dataloader, bs) : eachobs(dataloader)
            for (i, (x, y)) in enumerate(dataloader)
                step!(i, x, y, args...)

                !isnothing(progress) &&
                    @logprogress ((epoch - 1) * nobs(dataloader) + i) / (nepochs * nobs(dataloader))
            end
        end
    end
end

function run_dagger!(step!, data, args...; nepochs = 1, bs = 1, shuffle = true, progress = nothing)
    progressname = isnothing(progress) ? "" : progress
    @withprogress name=progressname begin
        for epoch in 1:nepochs
            @info "Starting epoch $epoch ..."

            dataloader = shuffle ? shuffleobs(data) : data
            dataloader = (bs > 1) ? eachbatch(dataloader, bs) : eachobs(dataloader)
            for (i, (x, y)) in enumerate(dataloader)
                blockers = step!(i, x, y, args...)
                foreach(b -> wait(b), blockers)

                !isnothing(progress) &&
                    @logprogress ((epoch - 1) * nobs(dataloader) + i) / (nepochs * nobs(dataloader))
            end
        end
    end
end

function run!(step!, ts::AbstractVector, args...)
    for t in ts
        step!(t, args...)
    end
end

function predict(net, xs, ys)
    ŷs = similar(ys, Float32)
    net = net |> cpu
    net_state = state(net)
    run!((xs, ys), net, net_state; shuffle = false, progress = "TEST") do i, x, _, net, net_state
        z = mean(last(net(net_state, x, t, Δt)) for t in 0:Δt:Δtsample)
        ŷs[:, i] .= z
    end

    return ŷs
end


function fitdecoder!(decoder, ŷs, ys; nepochs = 1000, batchsize = 32, opt = Momentum())
    loss(y, ŷ) = Flux.Losses.logitcrossentropy(decoder(ŷ), y)
    for _ in 1:nepochs
        Flux.train!(loss, Flux.params(decoder), eachbatch((ys, ŷs), batchsize), opt)
    end

    return decoder
end
fitdecoder(ŷs, ys; kwargs...) = fitdecoder!(Dense(size(ys, 1), size(ŷs, 1)), ŷs, ys; kwargs...)
