using CairoMakie
using CairoMakie: RGBA

include("../../src/experiments.jl")

# hardware target (cpu or gpu)
target = gpu

## PROBLEM PARAMETERS

τff = 5f-3 # LIF time constant for FF network
τr = 50f-3 # LIF time constant for reservoirs
λ = 1.7 # chaotic level
γ = 20f0 # HSIC balance parameter
η0 = 5f-3 # initial hsic learning rate
ηr = 1f-4 # initial reservoir learning rate
Nhidden = 1000 # size of reservoir
noise = 2.5f-1 # reservoir exploratory noise
τavg = 10f-3 # signal smoothing constant
Δt = 1f-3 # simulation time step
Nsamples = 100 # number of data samples
Δtsample = 50f-3 # time to present each data sample
bs = 10 # effective batch size
nepochs = 50 # number of epochs
Tinit = 50f0 # warmup time
Tpre = 500f0 # pretraining time

## DATA GENERATION

Dx = product_distribution([Uniform(-1f0, 1f0), Uniform(-1f0, 1f0)])
Din = 2
Dout = 1
ϕ(x) = [tanh(x[1]), x[2]]
W = [[1 -1]]
data, W = generatedata(Din, Nsamples; Dx = Dx)
data = (Float32.(data[1]), Float32.(data[2]))

## EXPERIMENT SETUP

recording, accuracy = dense_test(data, target;
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

fig = Figure(figsize = (600, 1200))

dataplt = fig[1:2, 1:2] = Axis(fig; title = "Data Distribution", xlabel = "x₁", ylabel = "x₂")
pts = classificationplot!(dataplt, data)
decisionboundary!(dataplt, W[1])
axislegend(dataplt, pts, ["Class 1", "Class 2"]; position = :rt)

wplt = fig[1, 3] = Axis(fig; title = "Weights", xlabel = "Time (sec)", ylabel = "Value")
lines!(wplt, recording.t, recording.w1 ./ recording.w2; label = "W₁ / W₂", color = :blue)
hlines!(wplt, [W[1][1] / W[1][2]]; label = "W₁ / W₂ (true)", color = :blue, linestyle = :dash)
xlims!(wplt, Tinit + Tpre, Tinit + Tpre + nepochs * Nsamples * Δtsample)
ylims!(wplt, W[1][1] / W[1][2] - 2, W[1][1] / W[1][2] + 2)
fig[2, 3] = Legend(fig, wplt; orientation = :horizontal, tellheight = true)

errplt = fig[3, :] = Axis(fig; title = "HSIC Objective",
                               xlabel = "Time (sec)",
                               ylabel = "Error")
lines!(errplt, recording.t, recording.lossx; label = "HSIC(X, Z)", color = RGBA(0, 1, 0, 0.5))
lines!(errplt, recording.t, recording.lossy; label = "HSIC(Y, Z)", color = RGBA(1, 0, 0, 0.5))
hsic1 = recording.lossx .- γ .* recording.lossy
lpf = LowPassFilter(Δtsample * 5, [0f0])
hsic1_lpf = [only(lpf(x, Δt)) for x in hsic1]
lines!(errplt, recording.t, hsic1;
       label = "HSIC(X, Z) - $γ HSIC(Y, Z)", color = RGBA(0, 0, 1, 0.5))
lines!(errplt, recording.t, hsic1_lpf;
       label = "Averaged Objective", color = :blue)
vlines!(errplt, [Tinit + Tpre]; color = :black, linestyle = :dash)
xlims!(errplt, Tinit + Tpre - 10, Tinit + Tpre + nepochs * Nsamples * Δtsample)
fig[4, :] = Legend(fig, errplt; orientation = :horizontal, tellheight = true)

Label(fig[1, 1, TopLeft()], "A",
      textsize = 24,
      padding = (0, 5, 5, 0),
      halign = :right)
Label(fig[1, 3, TopLeft()], "B",
      textsize = 24,
      padding = (0, 5, 5, 0),
      halign = :right)
Label(fig[3, 1, TopLeft()], "C",
      textsize = 24,
      padding = (0, 5, 5, 0),
      halign = :right)

CairoMakie.save("output/linear.pdf", fig)
