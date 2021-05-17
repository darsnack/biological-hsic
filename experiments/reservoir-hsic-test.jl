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
include("../src/lpf.jl")
include("../src/reservoir.jl")

# hardware target (cpu or gpu)
target = gpu

## PROBLEM PARAMETERS

τ = 50f-3 # LIF time constant
λ = 1.2 # chaotic level
τavg = 5f-3 # signal smoothing constant
Tinit = 50f0 # warmup time
Ttrain = 500f0 # training time
Ttest = 100f0 # testing time
Δt = 1f-3 # simulation time step
Nsamples = 100 # number of data samples
Δtsample = 50f-3 # time to present each data sample
# learning rate
η(t)::Float32 = (t > Tinit && t <= Tinit + Ttrain) ?
                    1f-4 / (1 + (t - Tinit) / 20f0) :
                    zero(Float32)

# network sizes
Nx = 2
Ny = 1
Nz = 2
Nin = Nx + Ny + Nz # needs to be >= 1 even if no input
Nhidden = 2000
Nout = 1

# input signal
X = rand(Float32, Nx, Nsamples) |> target
Y = rand(Float32, Ny, Nsamples) |> target
Z = rand(Float32, Nz, Nsamples) |> target
σx = estσ(X)
σy = estσ(Y)
Kx = [k_hsic(x, x̂; σ = σx) for x in eachcol(X), x̂ in eachcol(X)]
Ky = [k_hsic(y, ŷ; σ = σy) for y in eachcol(Y), ŷ in eachcol(Y)]
timetoidx(t) = (t < 0) ? 1 : (Int(round(t / Δtsample)) % Nsamples) + 1
function input(t)
    (t < 0) && return zeros(Float32, Nin) |> target
    i = timetoidx(t)

    return concatenate(X[:, i], Y[:, i], Z[:, i])
end

# true signal
function f(t)::Float32
    is = timetoidx.([t - i * Δtsample for i in 0:5])
    N = length(is)
    ξ = sum(Kx[p, q] - 2 * Ky[p, q] for p in is, q in is) ./ (N - 1)^2

    return ξ
end

## PROBLEM SETUP

reservoir = Reservoir{Float32}(Nin => Nout, Nhidden; λ = λ, τ = τ) |> target
learner = RMHebb(reservoir; η = η, τ = τavg) |> target

## RECORDING SETUP

recording = (t = Float32[], z = Float32[], zlpf = Float32[], wnorm = Float32[])

## STATE INITIALIZATION

state = ReservoirState(reservoir)

## WARMUP

@info "Starting warmup..."
@progress "INIT" for t in 0:Δt:(Tinit - Δt)
    step!(reservoir, state, input, t, Δt)
    push!(recording.t, t)
    push!(recording.z, cpu(state.z)[1])
    push!(recording.zlpf, cpu(learner.zlpf.f̄)[1])
    push!(recording.wnorm, norm(reservoir.Wout))
end

## TRAIN

@info "Starting training..."
@progress "TRAIN" for t in Tinit:Δt:(Tinit + Ttrain - Δt)
    step!(reservoir, state, learner, input, f, t, Δt)
    push!(recording.t, t)
    push!(recording.z, cpu(state.z)[1])
    push!(recording.zlpf, cpu(learner.zlpf.f̄)[1])
    push!(recording.wnorm, norm(reservoir.Wout))
end

## TEST

@info "Starting testing..."
@progress "TEST" for t in (Tinit + Ttrain):Δt:(Tinit + Ttrain + Ttest)
    step!(reservoir, state, input, t, Δt; explore = false)
    push!(recording.t, t)
    push!(recording.z, cpu(state.z)[1])
    push!(recording.zlpf, cpu(learner.zlpf.f̄)[1])
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
       label = "True Signal (HSIC Global Error)", color = :blue)
lines!(train_plt, recording.t[train_init_range], recording.z[train_init_range];
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
vlines!(train_plt, [Tinit]; linestyle = :dash, color = :red, label = "Training Onset")

test_init_plt = fig[1, 2] = Axis(fig; title = "Output (Start of Testing)",
                                      xlabel = "Time (t)",
                                      ylabel = "Signal")
lines!(test_init_plt, recording.t[test_init_range], ztrue[test_init_range];
       label = "True Signal (HSIC Global Error)", color = :blue)
lines!(test_init_plt, recording.t[test_init_range], recording.z[test_init_range];
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
hideydecorations!(test_init_plt; grid = false)

test_final_plt = fig[1, 3] = Axis(fig; title = "Output (End of Testing)",
                                       xlabel = "Time (t)",
                                       ylabel = "Signal")
lines!(test_final_plt, recording.t[test_final_range], ztrue[test_final_range];
       label = "True Signal (HSIC Global Error)", color = :blue)
lines!(test_final_plt, recording.t[test_final_range], recording.z[test_final_range];
       label = "Raw Readout", color = RGBA(0, 1, 0, 0.5))
hideydecorations!(test_final_plt; grid = false)

# ylims!(train_plt, (0, 1))
linkyaxes!(train_plt, test_init_plt, test_final_plt)

fig[2, :] = Legend(fig, train_plt; orientation = :horizontal, tellheight = true, nbanks = 2)

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

save("output/hsic-test.pdf", fig)
