using Adapt
using CUDA
CUDA.allowscalar(false)

using ProgressLogging
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using Distributions, Random
using LinearAlgebra
using CairoMakie
using AbstractPlotting: RGBA

include("../src/utils.jl")
include("../src/lpf.jl")
include("../src/reservoir.jl")

# hardware target (cpu or gpu)
target = cpu

## PROBLEM PARAMETERS

τ = 50f-3 # LIF time constant
λ = 1.2 # chaotic level
τavg = 5f-3 # signal smoothing constant
Tinit = 50f0 # warmup time
Ttrain = 400f0 # training time
Ttest = 100f0 # testing time
Δt = 1f-3 # simulation time step
# learning rate
η(t) = (t > Tinit && t <= Tinit + Ttrain) ?
              5f-4 / (1 + (t - Tinit) / 20f0) :
              zero(Float32)

# network sizes
Nin = 1 # needs to be >= 1 even if no input
Nhidden = 1000
Nout = 1

# true signal
f(t) = sin(2π * t)

## PROBLEM SETUP

reservoir = Reservoir{Float32}(Nin => Nout, Nhidden; λ = λ, τ = τ) |> target
learner = RMHebb(reservoir; η = η, τ = τavg) |> target

## RECORDING SETUP

recording = (t = Float32[], z = Float32[], zlpf = Float32[], wnorm = Float32[])

## STATE INITIALIZATION

state = ReservoirState(reservoir)
input(t) = target(zeros(Float32, Nin))

## WARMUP

@info "Starting warmup..."
@progress "INIT" for t in 0:Δt:(Tinit - Δt)
    step!(reservoir, state, input, t, Δt)
    push!(recording.t, t)
    push!(recording.z, state.z[1])
    push!(recording.zlpf, learner.zlpf.f̄[1])
    push!(recording.wnorm, norm(reservoir.Wout))
end

## TRAIN

@info "Starting training..."
@progress "TRAIN" for t in Tinit:Δt:(Tinit + Ttrain - Δt)
    step!(reservoir, state, learner, input, f, t, Δt)
    push!(recording.t, t)
    push!(recording.z, state.z[1])
    push!(recording.zlpf, learner.zlpf.f̄[1])
    push!(recording.wnorm, norm(reservoir.Wout))
end

## TEST

@info "Starting testing..."
@progress "TEST" for t in (Tinit + Ttrain):Δt:(Tinit + Ttrain + Ttest)
    step!(reservoir, state, input, t, Δt; explore = false)
    push!(recording.t, t)
    push!(recording.z, state.z[1])
    push!(recording.zlpf, learner.zlpf.f̄[1])
    push!(recording.wnorm, norm(reservoir.Wout))
end

## PLOT RESULTS

ztrue = f.(recording.t)

fig = Figure()

train_init_range = Int.(round(1/Δt) * ((Tinit - 2):Δt:(Tinit + 3)))
test_init_range = Int.(round(1/Δt) * ((Tinit + Ttrain):Δt:(Tinit + Ttrain + 5)))
test_final_range = Int.(round(1/Δt) * ((Tinit + Ttrain + Ttest - 5):Δt:(Tinit + Ttrain + Ttest)))

train_plt = fig[1, 1] = Axis(fig; title = "Output (Start of Training)",
                                  xlabel = "Time (t)",
                                  ylabel = "Signal")
lines!(train_plt, recording.t[train_init_range], recording.zlpf[train_init_range];
       label = "Filtered Readout", color = :green)
lines!(train_plt, recording.t[train_init_range], ztrue[train_init_range];
       label = "True Signal (sin(t))", color = :blue)
lines!(train_plt, recording.t[train_init_range], recording.z[train_init_range];
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
vlines!(train_plt, [Tinit]; linestyle = :dash, color = :red, label = "Training Onset")

test_init_plt = fig[1, 2] = Axis(fig; title = "Output (Start of Testing)",
                                      xlabel = "Time (t)",
                                      ylabel = "Signal")
lines!(test_init_plt, recording.t[test_init_range], ztrue[test_init_range];
       label = "True Signal (sin(t))", color = :blue)
lines!(test_init_plt, recording.t[test_init_range], recording.z[test_init_range];
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
hideydecorations!(test_init_plt; grid = false)

test_final_plt = fig[1, 3] = Axis(fig; title = "Output (End of Testing)",
                                       xlabel = "Time (t)",
                                       ylabel = "Signal")
lines!(test_final_plt, recording.t[test_final_range], ztrue[test_final_range];
       label = "True Signal (sin(t))", color = :blue)
lines!(test_final_plt, recording.t[test_final_range], recording.z[test_final_range];
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
hideydecorations!(test_final_plt; grid = false)

fig[2, :] = Legend(fig, train_plt; orientation = :horizontal, tellheight = true)

wplt = fig[3, :] = Axis(fig; title = "Readout Weight Norm",
                             xlabel = "Time (t)",
                             ylabel = "norm(Wout)")
lines!(wplt, recording.t, recording.wnorm; color = :blue)
hidexdecorations!(wplt; grid = false)
ηplt = fig[4, :] = Axis(fig; title = "Learning Rate",
                             xlabel = "Time (t)",
                             ylabel = "η(t)")
lines!(ηplt, recording.t, η.(recording.t); color = :blue)
linkxaxes!(wplt, ηplt)

save("output/sine-test.pdf", fig)
