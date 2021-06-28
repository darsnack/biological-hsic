function run!(step!, data, args...; nepochs = 1, bs = 1, shuffle = true, progress = nothing)
    progressname = isnothing(progress) ? "" : progress
    @withprogress name=progressname begin
        for epoch in 1:nepochs
            isnothing(progress) && @info "Starting epoch $epoch ..."

            dataloader = shuffle ? shuffleobs(data) : data
            dataloader = (bs > 1) ? eachbatch(dataloader, bs) : eachobs(data)
            for (i, (x, y)) in enumerate(dataloader)
                step!(i, x, y, args...)

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
