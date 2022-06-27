using ParameterSchedulers
using ParameterSchedulers: Scheduler
using DataFrames, CSV
using ArgParse

include("../../src/experiments/mlp_bp.jl")

# hardware target (cpu or gpu)
target = gpu

## PROBLEM PARAMETERS

raw_args = ArgParseSettings()
@add_arg_table! raw_args begin
    "--trials"
        help = "a vector of trial #s"
        default = [1]
        arg_type = Int
        nargs = '+'
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
    "--lr"
        help = "the initial learning rate"
        default = 1e-2
        arg_type = Float64
    "--progress-rate"
        help = "how often the progress bar is updated in # of iterations"
        default = 1
        arg_type = Int
end

args = parse_args(raw_args)

# create optimizers
opt = Momentum(args["lr"])
schedule = Step(opt.eta, 0.5, args["nepochs"] ÷ 4)

# create network architecture spec
Din = 32*32*3
Dout = length(args["classes"])
layer_config = [Din => 256,
                256 => 128,
                128 => Dout]

## EXPERIMENT

if !isdir("output/cifar10/bp_data")
    mkpath("output/cifar10/bp_data")
end
for trial in args["trials"]
    @info "STARTING TRIAL $trial"
    saved_results = DataFrame()
    global output = cifar10_bp_test(layer_config, target;
                                    opt = opt,
                                    schedule = schedule,
                                    bs = args["batchsize"],
                                    nepochs = args["nepochs"],
                                    validation_points = args["validation_points"],
                                    percent_samples = args["percent_samples"],
                                    classes = args["classes"],
                                    progressrate = args["progress-rate"])
    recording, results = output
    saved_results[!, "t"] = recording.t
    saved_results[!, "accuracies"] = recording.accuracies

    CSV.write("output/cifar10/bp_data/trial_$trial.csv", saved_results)
end
