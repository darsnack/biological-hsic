using ParameterSchedulers
using ParameterSchedulers: Scheduler
using DataFrames, CSV
using TensorBoardLogger
using ArgParse

include("../../src/experiments/cnn_bp.jl")

# output directory
outdir = "output/mnist/cnn_bp_data"

## PROBLEM PARAMETERS

raw_args = ArgParseSettings()
@add_arg_table! raw_args begin
    "--target"
        help = "device target (cpu or gpu)"
        default = "cpu"
        arg_type = String
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
    "--tb-run"
        help = "the name of the TensorBoard log run (or empty string for plain logging)"
        default = ""
        arg_type = String
end

args = parse_args(raw_args)

# create hardware target
target = (args["target"] == "cpu") ? cpu :
         (args["target"] == "gpu") ? gpu :
         error("Unrecognized device target $(args["target"])")

# create optimizers
opt = Momentum(args["lr"])
schedule = ParameterSchedulers.Constant(opt.eta)

# create network architecture spec
Din = 28*28
Dout = length(args["classes"])
conv_config = [1, 16, 32, 64]
param_config = fill((pad = SamePad(), stride = 1), length(conv_config))
dense_config = [32, Dout]

## EXPERIMENT

isdir(outdir) || mkpath(outdir)
for trial in args["trials"]
    @info "STARTING TRIAL $trial"
    logger = nothing
    GC.gc()
    logger = isempty(args["tb-run"]) ? global_logger() :
                                       TBLogger("tb_logs/$(args["tb-run"])-$trial", tb_overwrite)
    saved_results = DataFrame()
    global output = mnist_cnn_bp_test(conv_config, param_config, dense_config, target;
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
