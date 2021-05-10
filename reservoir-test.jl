using Adapt
using LoopVectorization
using CUDA
CUDA.allowscalar(false)

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using DifferentialEquations
using Distributions, Random
using LinearAlgebra
using CairoMakie

include("lpf.jl")
include("reservoir.jl")

## HARDWARE SETUP

cpu(x::AbstractArray) = x
cpu(x::CuArray) = adapt(Array, x)
gpu(x::AbstractArray) = adapt(CuArray, x)
gpu(x::CuArray) = x

target = cpu

## PROBLEM PARAMETERS ##

τavg = 5f-3 # signal smoothing constant
η = 5f-4 # learning rate
Ttrain = 100 # training time
Ttest = 100 # testing time
Δt = 1f-3 # simulation time step

# network sizes
Nin = 1 # needs to be >= 1 even if no input
Nhidden = 1000
Nout = 1

# true signal
f(t) = sin(2π * 0.1 * t)

## PROBLEM SETUP

reservoir = Reservoir{Float32}(Nin => Nout, Nhidden) |> target
learner = RFORCE(reservoir; η = η, τ = τavg) |> target

## ODE TRAIN

u0 = state(reservoir)
p = (input = t -> target(zeros(Float32, Nin)), cache = ReservoirCache(reservoir))
tspan = (0.0, Ttrain)

cb = let f = f, Δt = Δt, learner = learner
    PeriodicCallback(integrator -> learner(integrator, f, Δt), Δt;
                     initial_affect = false)
end
prob = ODEProblem{true}(reservoir, u0, tspan, p)
solve(prob, Euler();
      dt = Δt,
      callback = cb,
      save_everystep = false,
      progress = true, progress_steps = 5000)

## ODE TEST

u0 = state(reservoir)
p = (input = t -> target(zeros(Float32, Nin)), cache = ReservoirCache(reservoir))
tspan = (0.0, Ttest)

recording = SavedValues(Float32, typeof(p.cache.z))
cb = SavingCallback((u, t, integrator) -> copy(integrator.p.cache.z), recording;
                    saveat = 0:Δt:Ttest)
prob = ODEProblem{true}(reservoir, u0, tspan, p)
solve(prob, Euler();
      dt = Δt,
      callback = cb,
      save_everystep = false,
      progress = true, progress_steps = 5000)

## PLOT RESULTS

fig = Figure()
fig[1, 1] = Axis(fig; title = "Reservoir Learning sin(t)",
                      xlabel = "Time (t)",
                      ylabel = "Signal")
lines!(fig[1, 1], recording.t, recording.saveval; label = "Raw Readout")
lines!(fig[1, 1], recording.t, f.(recording.t);
       linestyle = :dash,
       label = "True Signal")
savefig("./test.pdf", fig)