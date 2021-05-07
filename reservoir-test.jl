using DifferentialEquations
using Distributions
using LinearAlgebra
using GLMakie

include("lpf.jl")
include("reservoir.jl")

## PROBLEM PARAMETERS ##

τavg = 5f-3 # signal smoothing constant
η = 5f-4 # learning rate
Ttrain = 100 # training time
Ttest = 500 # testing time
Δt = 1f-3 # simulation time step

# network sizes
Nin = 1 # needs to be >= 1 even if no input
Nhidden = 1000
Nout = 1

# true signal
f(t) = sin(t)

## PROBLEM SETUP

reservoir = Reservoir{Float32}(Nin => Nout, Nhidden)
learner = RFORCE(reservoir; η = η, τ = τavg)

## ODE SETUP

u0 = state(reservoir)
p = (input = t -> 0, cache = ReservoirCache(reservoir))
tspan = (0.0, Ttrain)

cb = let f = f, Δt = Δt, learner = learner
    PeriodicCallback(integrator -> learner(integrator, f, Δt), Δt;
                     initial_affect = false)
end
prob = ODEProblem{true}(reservoir, u0, tspan, p)
integrator = init(prob, Euler(); dt = Δt)

## TRAIN ##

solve!(integrator)

## TEST ##



# fig = Figure()
# fig[1, 1] = Axis(fig; title = "Solution to the linear ODE with a thick line",
#                       xlabel = "Time (t)",
#                       ylabel = "u(t) (in μm)")
# lines!(fig[1, 1], sol.t, sol.u; label = "My Thick Line!")
# lines!(fig[1, 1], sol.t, map(t -> 0.5 * exp(1.01 * t), sol.t);
#        linewidth = 3, linestyle = :dash,
#        label = "True Solution!")
# display(current_figure())