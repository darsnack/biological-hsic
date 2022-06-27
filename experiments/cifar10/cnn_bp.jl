using ParameterSchedulers
using ParameterSchedulers: Scheduler
using DataFrames, CSV
using TensorBoardLogger
using ArgParse

include("../../src/experiments/cnn_bp.jl")

# hardware target (cpu or gpu)
target = gpu
# output directory
outdir = "output/cifar10/cnn_bp_data"

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
        default = 1e-3
        arg_type = Float64
    "--wd"
        help = "weight decay rate"
        default = 5e-4
        arg_type = Float64
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
opt = WeightDecayChain(Momentum(args["lr"]), args["wd"])
schedule = ParameterSchedulers.Constant(opt.eta)

# create network architecture spec
Dout = length(args["classes"])
channel_config = [3,
                  64, BatchNorm(64, relu), Dropout(0.3; dims = 3),
                  64, BatchNorm(64, relu), MaxPool((2, 2)),
                  128, BatchNorm(128, relu), Dropout(0.4; dims = 3),
                  128, BatchNorm(128, relu), MaxPool((2, 2)),
                  256, BatchNorm(256, relu), Dropout(0.4; dims = 3),
                  256, BatchNorm(256, relu), Dropout(0.4; dims = 3),
                  256, BatchNorm(256, relu), MaxPool((2, 2)),
                  512, BatchNorm(512, relu), Dropout(0.4; dims = 3),
                  512, BatchNorm(512, relu), Dropout(0.4; dims = 3),
                  512, BatchNorm(512, relu), MaxPool((2, 2)),
                  512, BatchNorm(512, relu), Dropout(0.4; dims = 3),
                  512, BatchNorm(512, relu), Dropout(0.4; dims = 3),
                  512, BatchNorm(512, relu), MaxPool((2, 2)),
                  Dropout(0.5; dims = 3)]
param_config = vcat(repeat([(pad = SamePad(), stride = 1), nothing, nothing], 13), [nothing])
dense_config = [512, Dropout(0.5), Dout]

## EXPERIMENT

if !isdir(outdir)
    mkpath(outdir)
end
for trial in args["trials"]
    @info "STARTING TRIAL $trial"
    logger = nothing
    GC.gc()
    logger = isempty(args["tb-run"]) ? global_logger() :
                                       TBLogger("tb_logs/$(args["tb-run"])-$trial", tb_overwrite)
    saved_results = DataFrame()
    global output = cifar10_cnn_bp_test(channel_config, param_config, dense_config, target;
                                        opt = opt,
                                        schedule = schedule,
                                        bs = args["batchsize"],
                                        nepochs = args["nepochs"],
                                        validation_points = args["validation_points"],
                                        percent_samples = args["percent_samples"],
                                        classes = args["classes"],
                                        progressrate = args["progress-rate"],
                                        logger = logger)
    recording, results = output
    saved_results[!, "t"] = recording.t
    saved_results[!, "accuracies"] = recording.accuracies

    CSV.write(joinpath(outdir, "trial_$trial.csv"), saved_results)
end
