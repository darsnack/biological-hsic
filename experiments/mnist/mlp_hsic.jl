# using CairoMakie
# using CairoMakie: RGBA
using ParameterSchedulers
using ParameterSchedulers: Scheduler
using DataFrames, CSV
using TensorBoardLogger
using ArgParse

include("../../src/experiments/mlp_approx.jl")

# hardware target (cpu or gpu)
target = cpu

## PROBLEM PARAMETERS

raw_args = ArgParseSettings()
@add_arg_table! raw_args begin
    "--nthreads"
        help = "set equal to # of layers (or 1 for no threading)"
        default = 1
        arg_type = Int
    "--trials"
        help = "a vector of trial #s"
        default = [1]
        arg_type = Int
        nargs = '+'
    "--tau"
        help = "LIF time constant for FF network"
        default = 2f-3
        arg_type = Float32
    "--sigma_x"
        help = "Input similarity distance"
        default = 2f-1
        arg_type = Float32
    "--sigma_y"
        help = "Output similarity distance"
        default = 1f0
        arg_type = Float32
    "--sigma_zs"
        help = "Layer output similarity distances"
        default = [5f-1, 5f-1, 5f-1]
        arg_type = Float32
        nargs = 3
    "--gammas"
        help = "a vector of coefficients for the HSIC objective by layer"
        # default = [20f0, 50f0, 200f0, 500f0]
        default = [20f0, 50f0, 200f0]
        arg_type = Float32
        nargs = 3
    "--dt"
        help = "simulation time step"
        default = 1f-3
        arg_type = Float32
    "--dsample"
        help = "time to present each data sample"
        default = 20f-3
        arg_type = Float32
    "--batchsize"
        default = 32
        arg_type = Int
    "--nepochs"
        default = 1
        arg_type = Int
    "--validation_points"
        help = "epochs on which to run + record validation"
        default = [1, 5, ntuple(i -> i * 10, 100 ÷ 10)...]
        arg_type = Int
        nargs = '+'
    "--percent_samples"
        help = "the % of samples in the train dataset to utilize"
        default = 100
        arg_type = Int
    "--classes"
        help = "a vector of the MNIST classes to subsample"
        default = collect(0:9)
        arg_type = Int
        nargs = '*'
    "--lrs"
        help = "a vector of the initial learning rates by layer"
        default = [0.01, 0.01, 0.01]
        arg_type = Float64
        nargs = 3
    "--progress-rate"
        help = "how often the progress bar is updated in # of iterations"
        default = 1
        arg_type = Int
    "--tb-run"
        help = "the name of the TensorBoard log run (or empty string for plain logging)"
        default = ""
        arg_type = String
end

args = parse_args(raw_args)

# create optimizers
opts = map(Descent, args["lrs"])
schedules = map(opts) do o
    Step(o.eta, 0.5, args["nepochs"] ÷ 2)
    # Sequence(o.eta => 20,
    #          Exp(λ = o.eta, γ = 0.5) => args["nepochs"])
end

# create network architecture spec
Din = 28*28
Dout = length(args["classes"])
layer_config = [64, 32, Dout]

## EXPERIMENT

if !isdir("output/mnist/hsic_data")
    mkpath("output/mnist/hsic_data")
end
for trial in args["trials"]
    @info "STARTING TRIAL $trial"
    logger = nothing
    GC.gc()
    logger = isempty(args["tb-run"]) ? global_logger() :
                                       TBLogger("tb_logs/$(args["tb-run"])-$trial", tb_overwrite)
    saved_results = DataFrame()
    global output = mnist_mlp_approx_test(layer_config, target;
                                          τff = args["tau"],
                                          σx = args["sigma_x"],
                                          σy = args["sigma_y"],
                                          σzs = args["sigma_zs"],
                                          γs = args["gammas"],
                                          opts = opts,
                                          schedules = schedules,
                                          Δt = args["dt"],
                                          Δtsample = args["dsample"],
                                          bs = args["batchsize"],
                                          nepochs = args["nepochs"],
                                          validation_points = args["validation_points"],
                                          percent_samples = args["percent_samples"],
                                          classes = args["classes"],
                                          nthreads = args["nthreads"],
                                          progressrate = args["progress-rate"],
                                          logger = logger)
    recording, results = output
    saved_results[!, "t"] = recording.t
    saved_results[!, "accuracies"] = recording.accuracies
    for i in 1:length(layer_config)
        saved_results[!, "lossx_$i"] = recording.lossxs[i]
        saved_results[!, "lossy_$i"] = recording.lossys[i]
    end

    CSV.write("output/mnist/hsic_data/trial_$trial.csv", saved_results)
end

## PLOT RESULTS

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

## PLOT CLASS PREDICTIONS

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
