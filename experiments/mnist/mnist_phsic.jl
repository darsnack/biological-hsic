using CairoMakie
using CairoMakie: RGBA
using ParameterSchedulers
using ParameterSchedulers: Scheduler
using DataFrames, CSV

include("../../src/experiments.jl")

# hardware target (cpu or gpu)
target = cpu

## PROBLEM PARAMETERS

ntrials = 1
τff = 2f-3 # LIF time constant for FF network
γs = (2f0, 2f0, 2f0, 2f0) # HSIC balance parameter
Δt = 1f-3 # simulation time step
Δtsample = 20f-3 # time to present each data sample
nepochs = 75 # number of epochs
validation_points = (1, 5, 10, 20, 30, 40, 50, 60, 70)
percent_samples = 50 # percent of samples in the train dataset to retain
classes = 0:9 # subset of MNIST classes to use
opts = (Momentum(1e-6), Momentum(1e-6), Momentum(1e-6), Momentum(1e-6))
schedules = map(opts) do o
    Sequence(o.eta      => 20,
             o.eta / 10 => 30,
             o.eta / 20 => nepochs - 50)
end

## EXPERIMENT SETUP

Din = 28*28
Dout = length(classes)

layer_config = [Din => 64,
                64 => 32,
                32 => Dout,
                Dout => Dout]

## EXPERIMENT

for trial in 1:ntrials
    @info "STARTING TRIAL $trial"
    saved_results = DataFrame()
    global output = phsic_test(layer_config, target;
                               τff = τff,
                               γs = γs,
                               opts = opts,
                               schedules = schedules,
                               Δt = Δt,
                               Δtsample = Δtsample,
                               nepochs = nepochs,
                               validation_points = validation_points,
                               percent_samples = percent_samples,
                               classes = classes,
                               nthreads = length(layer_config))
    recording, results = output
    saved_results[!, "t"] = recording.t
    saved_results[!, "accuracies"] = recording.accuracies
    for i in 1:length(layer_config)
        saved_results[!, "lossx_$i"] = recording.lossxs[i]
        saved_results[!, "lossy_$i"] = recording.lossys[i]
    end

    CSV.write("output/phsic_data/trial_$trial.csv", saved_results)
end

# ## PLOT RESULTS

# fig = Figure()

# for i in 1:length(layer_config)
#     hsicerr = recording.lossxs[i] .- γs[i] .* recording.lossys[i]
#     hsicerr_avg = cumsum(hsicerr) ./ (1:length(hsicerr))

#     errplt = fig[i, 1] = Axis(fig; title = "HSIC Objective (Layer $i, γ = $(γs[i]))",
#                                    ylabel = "Components",
#                                    yaxisposition = :right)
#     lossplt = fig[i, 1] = Axis(fig; ylabel = "Loss", yticklabelcolor = :blue)
#     hsicx = lines!(errplt, recording.t, recording.lossxs[i]; color = RGBA(0, 1, 0, 0.5))
#     hsicy = lines!(errplt, recording.t, recording.lossys[i]; color = RGBA(1, 0, 0, 0.5))
#     lines!(lossplt, recording.t, hsicerr; color = RGBA(0, 0, 1, 0.5))
#     loss = lines!(lossplt, recording.t, hsicerr_avg; color = :blue)
#     tinit = vlines!(errplt, [Tinit]; color = :black, linestyle = :dash)
#     xlims!(errplt, Tinit - 10, recording.t[end])
#     xlims!(lossplt, Tinit - 10, recording.t[end])
#     hidespines!(errplt)
#     hidexdecorations!(errplt)

#     if i == length(layer_config)
#         lossplt.xlabel = "Time (sec)"
#         Legend(fig[i + 1, :],
#                [hsicx, hsicy, loss], ["HSIC(X, Z)", "HSIC(Y, Z)", "HSIC(X, Z) - γ HSIC(Y, Z)"];
#                orientation = :horizontal, tellheight = true, tellwidth = false)
#         # fig[i + 1, :] = Legend(fig, lossplt; orientation = :horizontal, tellheight = true, tellwidth = false)
#     else
#         hidexdecorations!(lossplt; grid = false, minorgrid = false)
#     end
# end

# CairoMakie.save("output/mnist-test.pdf", fig)

# ## PLOT CLASS PREDICTIONS

# fig = Figure()
# axs = []
# nrows = 2
# ncols = ceil(Int, length(classes) / 2)
# for i in 1:nrows, j in 1:ncols
#     ax = fig[i, j] = Axis(fig)
#     if i == nrows
#         ax.xlabel = "Output Indices"
#     else
#         hidexdecorations!(ax; grid = false, minorgrid = false)
#     end
#     if j == 1
#         ax.ylabel = "Output Value"
#     else
#         hideydecorations!(ax; grid = false, minorgrid = false)
#     end
#     push!(axs, ax)
# end

# ys = Flux.onecold(results.data.test[2], classes)
# ŷs = results.data.predictions
# classpredictionplot!(axs, ŷs, ys, classes)

# CairoMakie.save("output/mnist-class-predictions.pdf", fig)
