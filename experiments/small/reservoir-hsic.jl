using CairoMakie
using CairoMakie: RGBA
using Dates: now

include("../../src/setup.jl")
include("../../src/experiments/reservoir_test.jl")

# hardware target (cpu or gpu)
target = gpu
CUDA.device!(1) # adjust this to control which GPU is used

## PROBLEM PARAMETERS

η0 = 5f-5 # initial LR
γ = 2 # HSIC balance parameter
τ = 50f-3 # LIF time constant
λ = 1.2 # chaotic level
τavg = 5f-3 # signal smoothing constant
train_epochs = 100 # training time
test_epochs = 5 # testing time
Δt = 1f-3 # simulation time step
Δtsample = 20f-3 # time to present each data sample
bs = 6 # effective batch size
hidden_noise = 5f-6 # reservoir hidden noise
output_noise = 1f-2 # reservoir exploratory noise
Nsamples = 100 # number of data samples
Nhidden = 10_000 # number of hidden neurons in reservoir

# input data
Nx = 50
Ny = 1
Nz = 10
X = rand(Float32, Nx, Nsamples)
Y = rand(Float32, Ny, Nsamples)
Z = rand(Float32, Nz, Nsamples)

## EXPERIMENT

logger = WandbBackend(project = "biological-hsic", name = "reservoir-test-$(now())")
results = reservoir_test((X, Y, Z), Nhidden, target;
                         η0 = η0,
                         γ = γ,
                         τ = τ,
                         λ = λ,
                         τavg = τavg,
                         train_epochs = train_epochs,
                         test_epochs = test_epochs,
                         Δt = Δt,
                         Δtsample = Δtsample,
                         bs = bs,
                         hidden_noise = hidden_noise,
                         output_noise = output_noise,
                         logger = logger)
learner, training_scheme, validation_scheme = results

## CLEANUP

close(logger)

## STORE RESULTS

saved_results = DataFrame()
nsteps_per_epoch = Nsamples * ceil(Int, Δtsample / Δt)
saved_results[!, "t"] = 1:(Δt * nsteps_per_epoch * (train_epochs + test_epochs))
saved_results[!, "loss"] = 1:learner.cbstate.metricsstep["Loss"]

##

# fig = Figure()

# range_to_idx(ts) = Int.(round.(ts / Δt))
# train_init_ts = (Tinit - 2):Δt:(Tinit + 3)
# train_init_idx = range_to_idx(train_init_ts)
# test_init_ts = (Tinit + Ttrain):Δt:(Tinit + Ttrain + 5)
# test_init_idx = range_to_idx(test_init_ts)
# test_final_ts = (Tinit + Ttrain + Ttest - 5):Δt:(Tinit + Ttrain + Ttest)
# test_final_idx = range_to_idx(test_final_ts)

# train_plt1 = fig[1, 1] = Axis(fig; title = "Output 1 (Start of Training)",
#                                     xlabel = "Time (t)",
#                                     ylabel = "Signal")
# lines!(train_plt1, recording.t[train_init_idx], first.(recording.zlpf[train_init_idx]);
#         label = "Filtered Readout", color = :green)
# lines!(train_plt1, recording.t[train_init_idx], first.(recording.f[train_init_idx]);
#         label = "True Signal (HSIC Global Error)", color = :blue)
# lines!(train_plt1, recording.t[train_init_idx], first.(recording.z[train_init_idx]);
#         label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
# vlines!(train_plt1, [Tinit]; linestyle = :dash, color = :red, label = "Training Onset")

# test_init_plt1 = fig[1, 2] = Axis(fig; title = "Output 1 (Start of Testing)",
#                                     xlabel = "Time (t)",
#                                     ylabel = "Signal")
# lines!(test_init_plt1, recording.t[test_init_idx], first.(recording.f[test_init_idx]);
#         label = "True Signal (HSIC Global Error)", color = :blue)
# lines!(test_init_plt1, recording.t[test_init_idx], first.(recording.z[test_init_idx]);
#         label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
# hideydecorations!(test_init_plt1; grid = false, ticks = false, ticklabels = false)

# test_final_plt1 = fig[1, 3] = Axis(fig; title = "Output 1 (End of Testing)",
#                                     xlabel = "Time (t)",
#                                     ylabel = "Signal")
# lines!(test_final_plt1, recording.t[test_final_idx], first.(recording.f[test_final_idx]);
#         label = "True Signal (HSIC Global Error)", color = :blue)
# lines!(test_final_plt1, recording.t[test_final_idx], first.(recording.z[test_final_idx]);
#         label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
# hideydecorations!(test_final_plt1; grid = false)

# linkyaxes!(test_init_plt1, test_final_plt1)

# train_plt2 = fig[2, 1] = Axis(fig; title = "Output 2 (Start of Training)",
#                                     xlabel = "Time (t)",
#                                     ylabel = "Signal")
# lines!(train_plt2, recording.t[train_init_idx], last.(recording.zlpf[train_init_idx]);
#         label = "Filtered Readout", color = :green)
# lines!(train_plt2, recording.t[train_init_idx], last.(recording.f[train_init_idx]);
#         label = "True Signal (HSIC Global Error)", color = :blue)
# lines!(train_plt2, recording.t[train_init_idx], last.(recording.z[train_init_idx]);
#         label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
# vlines!(train_plt2, [Tinit]; linestyle = :dash, color = :red, label = "Training Onset")

# test_init_plt2 = fig[2, 2] = Axis(fig; title = "Output 2 (Start of Testing)",
#                                     xlabel = "Time (t)",
#                                     ylabel = "Signal")
# lines!(test_init_plt2, recording.t[test_init_idx], last.(recording.f[test_init_idx]);
#         label = "True Signal (HSIC Global Error)", color = :blue)
# lines!(test_init_plt2, recording.t[test_init_idx], last.(recording.z[test_init_idx]);
#         label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
# hideydecorations!(test_init_plt2; grid = false, ticks = false, ticklabels = false)

# test_final_plt2 = fig[2, 3] = Axis(fig; title = "Output 2 (End of Testing)",
#                                     xlabel = "Time (t)",
#                                     ylabel = "Signal")
# lines!(test_final_plt2, recording.t[test_final_idx], last.(recording.f[test_final_idx]);
#         label = "True Signal (HSIC Global Error)", color = :blue)
# lines!(test_final_plt2, recording.t[test_final_idx], last.(recording.z[test_final_idx]);
#         label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
# hideydecorations!(test_final_plt2; grid = false)

# linkyaxes!(test_init_plt2, test_final_plt2)

# fig[3, :] = Legend(fig, train_plt1; orientation = :horizontal, tellheight = true, nbanks = 2)
# # fig[4, :] = Legend(fig, train_plt2; orientation = :horizontal, tellheight = true, nbanks = 2)

# wplt = fig[4, :] = Axis(fig; title = "Readout Weight Norm",
#                             xlabel = "Time (t)",
#                             ylabel = "norm(Wout)")
# lines!(wplt, recording.t, recording.wnorm; color = :blue)

# CairoMakie.save("output/hsic-test.pdf", fig)

# ## PLOT CAMERA READY

# fig = Figure(figsize = (600, 1200))

# range_to_idx(ts) = Int.(round.(ts / Δt))
# train_init_ts = (Tinit - 2):Δt:(Tinit + 3)
# train_init_idx = range_to_idx(train_init_ts)
# test_init_ts = (Tinit + Ttrain):Δt:(Tinit + Ttrain + 5)
# test_init_idx = range_to_idx(test_init_ts)
# test_final_ts = (Tinit + Ttrain + Ttest - 5):Δt:(Tinit + Ttrain + Ttest)
# test_final_idx = range_to_idx(test_final_ts)

# train_plt1 = fig[1, 1] = Axis(fig; title = "Output (Start of Training)",
#                                     xlabel = "Time (t)",
#                                     ylabel = "Signal")
# lines!(train_plt1, recording.t[train_init_idx], first.(recording.f[train_init_idx]);
#         label = "True Signal (ξ)", color = :blue)
# lines!(train_plt1, recording.t[train_init_idx], first.(recording.zlpf[train_init_idx]);
#         label = "Filtered Readout", color = :green)
# lines!(train_plt1, recording.t[train_init_idx], first.(recording.z[train_init_idx]);
#         label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
# vlines!(train_plt1, [Tinit]; linestyle = :dash, color = :red, label = "Training Onset")
# hidexdecorations!(train_plt1; grid = false, ticks = false, ticklabels = false)

# test_init_plt1 = fig[2, 1] = Axis(fig; title = "Output (Start of Testing)",
#                                     xlabel = "Time (t)",
#                                     ylabel = "Signal")
# lines!(test_init_plt1, recording.t[test_init_idx], first.(recording.f[test_init_idx]);
#         label = "True Signal (ξ)", color = :blue)
# lines!(test_init_plt1, recording.t[test_init_idx], first.(recording.z[test_init_idx]);
#         label = "Raw Readout", color = RGBA(0, 1, 0, 0.8))
# hidexdecorations!(test_init_plt1; grid = false, ticks = false, ticklabels = false)

# test_final_plt1 = fig[3, 1] = Axis(fig; title = "Output (End of Testing)",
#                                     xlabel = "Time (t)",
#                                     ylabel = "Signal")
# lines!(test_final_plt1, recording.t[test_final_idx], first.(recording.f[test_final_idx]);
#         label = "True Signal (ξ)", color = :blue)
# lines!(test_final_plt1, recording.t[test_final_idx], first.(recording.z[test_final_idx]);
#         label = "Raw Readout", color = RGBA(0, 1, 0, 0.8))

# linkyaxes!(test_init_plt1, test_final_plt1)

# fig[:, 2] = Legend(fig, train_plt1)

# for (i, label) in enumerate(["A", "B", "C"])
#     Label(fig[i, 1, TopLeft()], label,
#         textsize = 24,
#         padding = (0, 5, 5, 0),
#         halign = :right)
# end

# CairoMakie.save("output/reservoir-hsic.pdf", fig)
