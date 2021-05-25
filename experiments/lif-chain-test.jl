using CairoMakie
using AbstractPlotting: RGBA

include("../src/experiments.jl")

# hardware target (cpu or gpu)
target = gpu

## PROBLEM PARAMETERS

τff = 5f-3 # LIF time constant for FF network
τr = 50f-3 # LIF time constant for reservoirs
λ = 1.7 # chaotic level
γ = 5f0 # HSIC balance parameter
η0 = 5f-3 # initial hsic learning rate
ηr = 1f-4 # initial reservoir learning rate
Nhidden = 1000 # size of reservoir
noise = 2.5f-1 # reservoir exploratory noise
τavg = 10f-3 # signal smoothing constant
Δt = 1f-3 # simulation time step
Nsamples = 100 # number of data samples
Δtsample = 50f-3 # time to present each data sample
bs = 10 # effective batch size
nepochs = 100 # number of epochs
Tinit = 50f0 # warmup time
Tpre = 500f0 # pretraining time

## DATA GENERATION

Dx = product_distribution([Uniform(-1f0, 1f0), Uniform(-1f0, 1f0)])
Din = 2
Dout = 1
ϕ(x) = [tanh(x[1]), x[2]]
W = [[1 -1]]
data, _ = generatedata(Din, Nsamples; Dx = Dx, ϕ = [ϕ])
data = (Float32.(data[1]), Float32.(data[2]))

## EXPERIMENT SETUP

layer_config = [Din => Din, Din => Dout]
recording = chain_test(data, layer_config, target;
                       τff = τff,
                       τr = τr,
                       λ = λ,
                       γ = γ,
                       η0 = η0,
                       ηr = ηr,
                       Nhidden = Nhidden,
                       noise = noise,
                       τavg = τavg,
                       Δt = Δt,
                       Nsamples = Nsamples,
                       Δtsample = Δtsample,
                       bs = bs,
                       nepochs = nepochs,
                       Tinit = Tinit,
                       Tpre = Tpre)

## PLOT RESULTS

fig = Figure()

dataplt = fig[1, 1:2] = Axis(fig; title = "Data Distribution", xlabel = "x₁", ylabel = "x₂")
pts = classificationplot!(dataplt, data)
decisionboundary!(dataplt, W[1]; ϕ = ϕ)
axislegend(dataplt, pts, ["Class 1", "Class 2"]; position = :rt)

errplt1 = fig[2, 1] = Axis(fig; title = "HSIC Objective (Layer 1)",
                                xlabel = "Time (sec)",
                                ylabel = "Error")
lines!(errplt1, recording.t, recording.lossxs[1]; label = "HSIC(X, Z)", color = :green)
lines!(errplt1, recording.t, recording.lossys[1]; label = "HSIC(Y, Z)", color = :red)
lines!(errplt1, recording.t, recording.lossxs[1] .- γ .* recording.lossys[1];
       label = "HSIC(X, Z) - $γ HSIC(Y, Z)", color = :blue)
vlines!(errplt1, [Tinit + Tpre]; color = :black, linestyle = :dash)
xlims!(errplt1, Tinit + Tpre - 10, Tinit + Tpre + nepochs * Nsamples * Δtsample)

errplt2 = fig[2, 2] = Axis(fig; title = "HSIC Objective (Layer 2)",
                                xlabel = "Time (sec)",
                                ylabel = "Error")
lines!(errplt2, recording.t, recording.lossxs[2]; label = "HSIC(X, Z)", color = :green)
lines!(errplt2, recording.t, recording.lossys[2]; label = "HSIC(Y, Z)", color = :red)
lines!(errplt2, recording.t, recording.lossxs[2] .- γ .* recording.lossys[2];
       label = "HSIC(X, Z) - $γ HSIC(Y, Z)", color = :blue)
vlines!(errplt2, [Tinit + Tpre]; color = :black, linestyle = :dash)
xlims!(errplt2, Tinit + Tpre - 10, Tinit + Tpre + nepochs * Nsamples * Δtsample)
fig[3, :] = Legend(fig, errplt2; orientation = :horizontal, tellheight = true)

save("output/lif-chain-test.pdf", fig)
