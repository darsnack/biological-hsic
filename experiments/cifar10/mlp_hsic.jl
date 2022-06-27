using ParameterSchedulers
using ParameterSchedulers: Scheduler
using DataFrames, CSV
using ArgParse

include("../../src/experiments/mlp_approx.jl")

# hardware target (cpu or gpu)
target = gpu

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
    "--gammas"
        help = "a vector of coefficients for the HSIC objective by layer"
        # default = [20f0, 50f0, 200f0, 500f0]
        default = [5f0, 10f0, 200f0]
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
        default = [7.5e-1 for _ in 1:3]
        arg_type = Float64
        nargs = 3
    "--progress-rate"
        help = "how often the progress bar is updated in # of iterations"
        default = 1
        arg_type = Int
end

args = parse_args(raw_args)

# create optimizers
opts = map(Descent, args["lrs"])
schedules = map(opts) do o
    Step(o.eta, 0.5, args["nepochs"] ÷ 4)
end

# create network architecture spec
Din = 32*32*3
Dout = length(args["classes"])
layer_config = [Din => 256,
                256 => 128,
                128 => Dout]

## EXPERIMENT

if !isdir("output/cifar10/hsic_data")
    mkpath("output/cifar10/hsic_data")
end
for trial in args["trials"]
    @info "STARTING TRIAL $trial"
    saved_results = DataFrame()
    global output = cifar10_approx_test(layer_config, target;
                                        τff = args["tau"],
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
                                        progressrate = args["progress-rate"])
    recording, results = output
    saved_results[!, "t"] = recording.t
    saved_results[!, "accuracies"] = recording.accuracies
    for i in 1:length(layer_config)
        saved_results[!, "lossx_$i"] = recording.lossxs[i]
        saved_results[!, "lossy_$i"] = recording.lossys[i]
    end

    CSV.write("output/cifar10/hsic_data/trial_$trial.csv", saved_results)
end
