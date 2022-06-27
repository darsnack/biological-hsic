using CairoMakie
using CairoMakie: RGBA
using ParameterSchedulers
using ParameterSchedulers: Scheduler
using DataFrames, CSV

include("../../src/experiments.jl")

# hardware target (cpu or gpu)
target = cpu

## SWEEP PARAMETERS

validation_points = 5:5:20
batch_sizes = (2, 4, 6, 10, 16, 32)

## PROBLEM PARAMETERS

τff = 2f-3 # LIF time constant for FF network
γs = (50f0, 100f0, 200f0, 500f0) # HSIC balance parameter
Δt = 1f-3 # simulation time step
Δtsample = 20f-3 # time to present each data sample
nepochs = last(validation_points)
percent_samples = 50 # percent of samples in the train dataset to retain
classes = 0:9 # subset of MNIST classes to use
ηs = (1e-4, 1e-3, 1e-3, 5e-3, 5e-3, 5e-2)

## EXPERIMENT SETUP

Din = 28*28
Dout = length(classes)

layer_config = [Din => 64,
                64 => 32,
                32 => Dout,
                Dout => Dout]

## EXPERIMENT

N = length(validation_points)
M = length(batch_sizes)
saved_results = DataFrame(batch_size = zeros(M*N),
                          nepochs = zeros(M*N),
                          accuracy = Array{Union{Missing, Float32}, 1}(undef, M*N))
for (i, bs) in enumerate(batch_sizes)
    @info "STARTING TRIAL (bs = $bs)"
    global saved_results
    global output = mnist_test(layer_config, target;
                               τff = τff,
                               γs = γs,
                               opts = [Momentum(ηs[i]) for _ in 1:length(layer_config)],
                               Δt = Δt,
                               Δtsample = Δtsample,
                               bs = bs,
                               nepochs = nepochs,
                               validation_points = validation_points,
                               percent_samples = percent_samples,
                               classes = classes,
                               nthreads = length(layer_config))
    recording, results = output
    saved_results[((i - 1) * N + 1):((i - 1) * N + N), "batch_size"] .= bs
    saved_results[((i - 1) * N + 1):((i - 1) * N + N), "nepochs"] .= validation_points
    saved_results[((i - 1) * N + 1):((i - 1) * N + N), "accuracy"] .= collect(skipmissing(recording.accuracies))[1:N]

    CSV.write("output/memory_sweep/bs_$bs.csv", saved_results)
end

## PLOT RESULTS

fig = Figure()

accuracies = Float32.(permutedims(reshape(saved_results[!, "accuracy"], N, M)))
accuracies .= accuracies ./ maximum(accuracies)

plt = fig[1, 1] = Axis(fig; title = "Final Normalized Test Accuracy on MNIST",
                            xlabel = "Effective Batch Size",
                            ylabel = "# of Epochs")
hmap = heatmap!(plt, accuracies)
Colorbar(fig[1, 2], hmap)
plt.xticks = (1:M, string.(collect(batch_sizes)))
plt.yticks = (1:N, string.(validation_points))

CairoMakie.save("output/memory-sweep.pdf", fig)
