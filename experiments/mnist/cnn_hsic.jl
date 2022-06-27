using ParameterSchedulers
using ParameterSchedulers: Scheduler
using DataFrames, CSV
using TensorBoardLogger
using ArgParse

include("../../src/experiments/cnn_approx.jl")

# output directory
outdir = "output/mnist/cnn_approx_data"

## PROBLEM PARAMETERS

raw_args = ArgParseSettings()
@add_arg_table! raw_args begin
    "--target"
        help = "device target (cpu or gpu)"
        default = "cpu"
        arg_type = String
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
        default = [5f-1, 2f-1, 2f-1]
        arg_type = Float32
        nargs = 3
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
        default = [7.5e-1, 2.5e-1, 2.5e-1]
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

# create hardware target
target = (args["target"] == "cpu") ? cpu :
         (args["target"] == "gpu") ? gpu :
         error("Unrecognized device target $(args["target"])")

# create optimizers
opts = map(Descent, args["lrs"])
schedules = map(opts) do o
    Step(o.eta, 0.5, args["nepochs"] ÷ 4)
    # Sequence(o.eta => 20,
    #          Exp(λ = o.eta, γ = 0.5) => args["nepochs"])
end

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
    global output = mnist_cnn_approx_test(conv_config, param_config, dense_config, target;
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

    CSV.write(joinpath(outdir, "trial_$trial.csv"), saved_results)
end
