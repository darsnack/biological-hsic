using CairoMakie
using CairoMakie: RGBA

include("../../src/experiments.jl")

# hardware target (cpu or gpu)
target = gpu

## PROBLEM PARAMETERS

τff = 5f-3 # LIF time constant for FF network
τr = 5f-3 # LIF time constant for reservoirs
λ = 1.7 # chaotic level
γ = 50f0 # HSIC balance parameter
η0 = 5f-3 # initial hsic learning rate
ηr = 1f-4 # initial reservoir learning rate
Nhidden = 1000 # size of reservoir
noise = 2.5f-1 # reservoir exploratory noise
τavg = 10f-3 # signal smoothing constant
Δt = 1f-3 # simulation time step
Nsamples = 100 # number of data samples
Δtsample = 50f-3 # time to present each data sample
bs = 10 # effective batch size
nepochs = 20 # number of epochs
Tinit = 50f0 # warmup time
Tpre = 500f0 # pretraining time

## DATA GENERATION

Dx = product_distribution([Uniform(-1f0, 1f0), Uniform(-1f0, 1f0)])
Din = 2
Dout = 1
ϕ(x) = [tanh(3 * x[1]), x[2]]
data, W = generatedata(Din, Nsamples; Dx = Dx, W = [[1 -1]], ϕ = [ϕ])
data = (Float32.(data[1]), Float32.(data[2]))

## EXPERIMENT SETUP

layer_config = [Din => Din, Din => Dout]
recording, accuracy = chain_test(data, layer_config, target;
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
                                 Δtsample = Δtsample,
                                 bs = bs,
                                 nepochs = nepochs,
                                 Tinit = Tinit,
                                 Tpre = Tpre)

## PLOT RESULTS

fig = Figure(figsize = (600, 1200))

dataplt = fig[1, 1:2] = Axis(fig; title = "Data Distribution", xlabel = "x₁", ylabel = "x₂")
pts = classificationplot!(dataplt, data)
decisionboundary!(dataplt, W[1]; ϕ = ϕ)
axislegend(dataplt, pts, ["Class 1", "Class 2"]; position = :rt)

errplt1 = fig[2, 1] = Axis(fig; title = "HSIC Objective (Layer 1)",
                                xlabel = "Time (sec)",
                                ylabel = "Error")
lpf = LowPassFilter(Δtsample * 5, [0f0])
hsic1 = recording.lossxs[1] .- γ .* recording.lossys[1]
hsic1_lpf = [only(lpf(x, Δt)) for x in hsic1]
lines!(errplt1, recording.t, recording.lossxs[1]; label = "HSIC(X, Z)", color = RGBA(0, 1, 0, 0.5))
lines!(errplt1, recording.t, recording.lossys[1]; label = "HSIC(Y, Z)", color = RGBA(1, 0, 0, 0.5))
lines!(errplt1, recording.t, hsic1;
       label = "HSIC(X, Z) - $γ HSIC(Y, Z)", color = RGBA(0, 0, 1, 0.5))
lines!(errplt1, recording.t, hsic1_lpf;
       label = "Filtered Objective", color = :blue)
vlines!(errplt1, [Tinit + Tpre]; color = :black, linestyle = :dash)
xlims!(errplt1, Tinit + Tpre - 10, Tinit + Tpre + nepochs * Nsamples * Δtsample)

errplt2 = fig[2, 2] = Axis(fig; title = "HSIC Objective (Layer 2)",
                                xlabel = "Time (sec)",
                                ylabel = "Error")
lpf.f̄ .= 0f0
hsic2 = recording.lossxs[2] .- γ .* recording.lossys[2]
hsic2_lpf = [only(lpf(x, Δt)) for x in hsic1]
lines!(errplt2, recording.t, recording.lossxs[2]; label = "HSIC(X, Z)", color = RGBA(0, 1, 0, 0.5))
lines!(errplt2, recording.t, recording.lossys[2]; label = "HSIC(Y, Z)", color = RGBA(1, 0, 0, 0.5))
lines!(errplt2, recording.t, hsic2;
       label = "HSIC(X, Z) - $γ HSIC(Y, Z)", color = RGBA(0, 0, 1, 0.5))
lines!(errplt2, recording.t, hsic2_lpf;
       label = "Filtered Objective", color = :blue)
vlines!(errplt2, [Tinit + Tpre]; color = :black, linestyle = :dash)
xlims!(errplt2, Tinit + Tpre - 10, Tinit + Tpre + nepochs * Nsamples * Δtsample)
fig[3, :] = Legend(fig, errplt2; orientation = :horizontal, tellheight = true)

Label(fig[1, 1, TopLeft()], "A",
      textsize = 24,
      padding = (0, 5, 5, 0),
      halign = :right)
Label(fig[2, 1, TopLeft()], "B",
      textsize = 24,
      padding = (0, 5, 5, 0),
      halign = :right)

CairoMakie.save("output/tanh.pdf", fig)
