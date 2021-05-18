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
include("../src/hsic.jl")
include("../src/learning.jl")
include("../src/lpf.jl")
include("../src/reservoir.jl")

# hardware target (cpu or gpu)
target = gpu

## PROBLEM PARAMETERS

const τ = 50f-3 # LIF time constant
const λ = 1.7 # chaotic level
const τavg = 10f-3 # signal smoothing constant
const Tinit = 50f0 # warmup time
const Ttrain = 1000f0 # training time
const Ttest = 100f0 # testing time
const Δt = 1f-3 # simulation time step
const Nsamples = 100 # number of data samples
const Δtsample = 50f-3 # time to present each data sample
const bs = 6 # effective batch size
# learning rate
η(t)::Float32 = (t > Tinit && t <= Tinit + Ttrain) ?
                    1f-4 / (1 + (t - Tinit) / 20f0) :
                    zero(Float32)

# network sizes
const Nx = 2
const Ny = 1
const Nz = 2
const Nin = Nx + Ny + Nz # needs to be >= 1 even if no input
const Nhidden = 2000
const Nout = 1

# input data
const X = rand(Float32, Nx, Nsamples) |> target
const Y = rand(Float32, Ny, Nsamples) |> target
const Z = rand(Float32, Nz, Nsamples) |> target
const σx = estσ(X)
const σy = estσ(Y)
const σz = estσ(Z)
const Kx = [k_hsic(x, x̂; σ = σx) for x in eachcol(X), x̂ in eachcol(X)]
const Ky = [k_hsic(y, ŷ; σ = σy) for y in eachcol(Y), ŷ in eachcol(Y)]
const Kz = [k_hsic(z, ẑ; σ = σz) for z in eachcol(Z), ẑ in eachcol(Z)]

# input signal
timetoidx(t) = (t < 0) ? 1 : (Int(round(t / Δtsample)) % Nsamples) + 1
function input(t)
    (t < 0) && return zeros(Float32, Nin) |> target
    i = timetoidx(t)

    return concatenate(X[:, i], Y[:, i], Z[:, i])
end

# true signal
ξ = GlobalError{Float32}(bs, Nz)
function f(t)::Vector{Float32}
    is = timetoidx.([t - i * Δtsample for i in 0:(bs - 1)])
    kx = view(Kx, is, is)
    ky = view(Ky, is, is)
    kz = view(Kz, is, is)
    z = cpu(Z[:, is])

    return ξ(kx, ky, kz, z)
end

## PROBLEM SETUP

reservoir = Reservoir{Float32}(Nin => Nout, Nhidden; λ = λ, τ = τ, noiselevel = 2.5f-1) |> target
learner = RMHebb(reservoir; η = η, τ = τavg) |> target

## RECORDING SETUP

recording = (t = Float32[], z = Vector{Float32}[], zlpf = Vector{Float32}[], wnorm = Float32[])

## STATE INITIALIZATION

state = ReservoirState(reservoir)

## WARMUP

@info "Starting warmup..."
@progress "INIT" for t in 0:Δt:(Tinit - Δt)
    step!(reservoir, state, input, t, Δt)
    push!(recording.t, t)
    push!(recording.z, cpu(state.z))
    push!(recording.zlpf, cpu(learner.zlpf.f̄))
    push!(recording.wnorm, norm(reservoir.Wout))
end

## TRAIN

@info "Starting training..."
@progress "TRAIN" for t in Tinit:Δt:(Tinit + Ttrain - Δt)
    step!(reservoir, state, learner, input, f, t, Δt)
    push!(recording.t, t)
    push!(recording.z, cpu(state.z))
    push!(recording.zlpf, cpu(learner.zlpf.f̄))
    push!(recording.wnorm, norm(reservoir.Wout))
end

## TEST

@info "Starting testing..."
@progress "TEST" for t in (Tinit + Ttrain):Δt:(Tinit + Ttrain + Ttest)
    step!(reservoir, state, input, t, Δt; explore = false)
    push!(recording.t, t)
    push!(recording.z, cpu(state.z))
    push!(recording.zlpf, cpu(learner.zlpf.f̄))
    push!(recording.wnorm, norm(reservoir.Wout))
end

## PLOT RESULTS

fig = Figure()

range_to_idx(ts) = Int.(round.(ts / Δt))
train_init_ts = (Tinit - 2):Δt:(Tinit + 3)
train_init_idx = range_to_idx(train_init_ts)
test_init_ts = (Tinit + Ttrain):Δt:(Tinit + Ttrain + 5)
test_init_idx = range_to_idx(test_init_ts)
test_final_ts = (Tinit + Ttrain + Ttest - 5):Δt:(Tinit + Ttrain + Ttest)
test_final_idx = range_to_idx(test_final_ts)

train_plt1 = fig[1, 1] = Axis(fig; title = "Output 1 (Start of Training)",
                                   xlabel = "Time (t)",
                                   ylabel = "Signal")
lines!(train_plt1, recording.t[train_init_idx], first.(recording.zlpf[train_init_idx]);
       label = "Filtered Readout", color = :green)
lines!(train_plt1, recording.t[train_init_idx], first.(f.(train_init_ts));
       label = "True Signal (HSIC Global Error)", color = :blue)
lines!(train_plt1, recording.t[train_init_idx], first.(recording.z[train_init_idx]);
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
vlines!(train_plt1, [Tinit]; linestyle = :dash, color = :red, label = "Training Onset")

test_init_plt1 = fig[1, 2] = Axis(fig; title = "Output 1 (Start of Testing)",
                                       xlabel = "Time (t)",
                                       ylabel = "Signal")
lines!(test_init_plt1, recording.t[test_init_idx], first.(f.(test_init_ts));
       label = "True Signal (HSIC Global Error)", color = :blue)
lines!(test_init_plt1, recording.t[test_init_idx], first.(recording.z[test_init_idx]);
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
hideydecorations!(test_init_plt1; grid = false)

test_final_plt1 = fig[1, 3] = Axis(fig; title = "Output 1 (End of Testing)",
                                        xlabel = "Time (t)",
                                        ylabel = "Signal")
lines!(test_final_plt1, recording.t[test_final_idx], first.(f.(test_final_ts));
       label = "True Signal (HSIC Global Error)", color = :blue)
lines!(test_final_plt1, recording.t[test_final_idx], first.(recording.z[test_final_idx]);
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
hideydecorations!(test_final_plt1; grid = false)

linkyaxes!(train_plt1, test_init_plt1, test_final_plt1)

train_plt2 = fig[2, 1] = Axis(fig; title = "Output 2 (Start of Training)",
                                   xlabel = "Time (t)",
                                   ylabel = "Signal")
lines!(train_plt2, recording.t[train_init_idx], last.(recording.zlpf[train_init_idx]);
       label = "Filtered Readout", color = :green)
lines!(train_plt2, recording.t[train_init_idx], last.(f.(train_init_ts));
       label = "True Signal (HSIC Global Error)", color = :blue)
lines!(train_plt2, recording.t[train_init_idx], last.(recording.z[train_init_idx]);
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
vlines!(train_plt2, [Tinit]; linestyle = :dash, color = :red, label = "Training Onset")

test_init_plt2 = fig[2, 2] = Axis(fig; title = "Output 2 (Start of Testing)",
                                       xlabel = "Time (t)",
                                       ylabel = "Signal")
lines!(test_init_plt2, recording.t[test_init_idx], last.(f.(test_init_ts));
       label = "True Signal (HSIC Global Error)", color = :blue)
lines!(test_init_plt2, recording.t[test_init_idx], last.(recording.z[test_init_idx]);
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
hideydecorations!(test_init_plt2; grid = false)

test_final_plt2 = fig[2, 3] = Axis(fig; title = "Output 2 (End of Testing)",
                                        xlabel = "Time (t)",
                                        ylabel = "Signal")
lines!(test_final_plt2, recording.t[test_final_idx], last.(f.(test_final_ts));
       label = "True Signal (HSIC Global Error)", color = :blue)
lines!(test_final_plt2, recording.t[test_final_idx], last.(recording.z[test_final_idx]);
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
hideydecorations!(test_final_plt2; grid = false)

linkyaxes!(train_plt2, test_init_plt2, test_final_plt2)

fig[3, :] = Legend(fig, train_plt1; orientation = :horizontal, tellheight = true, nbanks = 2)
# fig[4, :] = Legend(fig, train_plt2; orientation = :horizontal, tellheight = true, nbanks = 2)

wplt = fig[4, :] = Axis(fig; title = "Readout Weight Norm",
                             xlabel = "Time (t)",
                             ylabel = "norm(Wout)")
lines!(wplt, recording.t, recording.wnorm; color = :blue)

save("output/hsic-test.pdf", fig)
