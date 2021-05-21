using Adapt
using LoopVectorization
using CUDA
CUDA.allowscalar(false)

using ProgressLogging
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using Distributions, Random
using DataStructures
using LinearAlgebra
using TensorCast
using Flux
using CairoMakie
using AbstractPlotting: RGBA

include("../src/utils.jl")
include("../src/hsic.jl")
include("../src/lpf.jl")
include("../src/reservoir.jl")
include("../src/networks.jl")
include("../src/data.jl")
include("../src/learning.jl")

# hardware target (cpu or gpu)
target = gpu

## PROBLEM PARAMETERS

τff = 5f-3 # LIF time constant for FF network
τr = 50f-3 # LIF time constant for reservoirs
λ = 1.7 # chaotic level
γ = 2f0 # HSIC balance parameter
τavg = 10f-3 # signal smoothing constant
Δt = 1f-3 # simulation time step
Nsamples = 100 # number of data samples
Δtsample = 50f-3 # time to present each data sample
bs = 15 # effective batch size
nepochs = 10 # number of epochs
Tinit = 50f0 # warmup time
Ttrain = Nsamples * Δtsample # training time
Ttest = Nsamples * Δtsample # testing time
η(t)::Float32 = (t > Tinit) ? 1f-4 / (1 + (t - Tinit) / 25f0) : zero(Float32)

## DATA GENERATION

Dx = product_distribution([Uniform(-1f0, 1f0), Uniform(-1f0, 1f0)])
Din = 2
Dout = 1
data, W = generatedata(Din, Nsamples; Dx = Dx)
data = (Float32.(data[1]), Float32.(data[2]))

## NETWORK SETUP

xencoder = RateEncoder(data[1], Δtsample)
yencoder = RateEncoder(data[2], Δtsample)

net = LIFDense{Float32}(Din, Dout; τ = τff)
netstate = state(net)

learner = HSICIdeal(xencoder, yencoder, net, netstate, bs;
                    λ = γ, nbuffer = Int(cld(Δtsample, Δt)))

## RECORDING SETUP

recording = (t = Float32[], lossx = Float32[], lossy = Float32[], w1 = Float32[], w2 = Float32[])
record_rate = Int(cld(Δtsample, Δt))
H = I - fill(1 / bs, bs, bs) # centering matrix

## WARMUP

@info "Starting warmup..."
@progress "INIT" for t in 0:Δt:(Tinit - Δt)
    # evaluate model
    x = xencoder(t)
    y = yencoder(t)
    z = net(netstate, x, t, Δt)

    # get weight update
    Δw = learner(x, y, x, copy(z))

    # record values
    if cld(t, Δt) % record_rate == 0
        push!(recording.t, t)
        push!(recording.lossx, tr(learner.Kx * H * learner.Kz * H) / (bs - 1)^2)
        push!(recording.lossy, tr(learner.Ky * H * learner.Kz * H) / (bs - 1)^2)
        push!(recording.w1, net.W[1])
        push!(recording.w2, net.W[2])
    end
end

## TRAINING

@info "Starting training..."
@progress "TRAIN" for epoch in 1:nepochs
    # shuffle data
    idx = shuffle(1:Nsamples)
    xencoder.data .= xencoder.data[:, idx]
    yencoder.data .= yencoder.data[:, idx]
    for t in (Tinit + (epoch - 1) * Ttrain):Δt:(Tinit + epoch * Ttrain - Δt)
        # evaluate model
        x = xencoder(t)
        y = yencoder(t)
        z = net(netstate, x, t, Δt)

        # get weight update
        Δw = learner(x, y, x, copy(z))
        net.W .+= η(t) * Δw

        # record values
        if cld(t, Δt) % record_rate == 0
            push!(recording.t, t)
            push!(recording.lossx, tr(learner.Kx * H * learner.Kz * H) / (bs - 1)^2)
            push!(recording.lossy, tr(learner.Ky * H * learner.Kz * H) / (bs - 1)^2)
            push!(recording.w1, net.W[1])
            push!(recording.w2, net.W[2])
        end
    end
end

## POST-TRAIN

@info "Starting testing..."
Ŷs = similar(data[2])
@progress "TEST" for (i, (x, y)) in enumerate(zip(eachcol(data[1]), eachcol(data[2])))
    for t in 0:Δt:Δtsample
        z = net(netstate, x, t, Δt)
        Ŷs[:, i] .= z
    end
end

@info "Starting decoding..."
decoder = Dense(Dout, Dout)
opt = Momentum(1e-3)
loss(y, ŷ) = Flux.Losses.logitbinarycrossentropy(y, decoder(ŷ))
for _ in 1:1000
    Flux.train!(loss, Flux.params(decoder), [(data[2], Ŷs)], opt)
end

acc = mean(round.(Flux.sigmoid.(decoder(Ŷs))) .== data[2])
@info "Accuracy = $(acc * 100)%"

## PLOT RESULTS

fig = Figure()

dataplt = fig[1, 1] = Axis(fig; title = "Data Distribution", xlabel = "x₁", ylabel = "x₂")
pts = classificationplot!(dataplt, data)
decisionboundary!(dataplt, W[1])
axislegend(dataplt, pts, ["Class 1", "Class 2"]; position = :rt)

wplt = fig[1, 2] = Axis(fig; title = "Weights", xlabel = "Time (sec)", ylabel = "Value")
lines!(wplt, recording.t, recording.w1; label = "W₁", color = :blue)
lines!(wplt, recording.t, recording.w2; label = "W₂", color = :green)
hlines!(wplt, [W[1][1]]; label = "W₁ (true)", color = :blue, linestyle = :dash)
hlines!(wplt, [W[1][2]]; label = "W₂ (true)", color = :green, linestyle = :dash)
xlims!(wplt, Tinit, Tinit + nepochs * Ttrain)
fig[1, 3] = Legend(fig, wplt)

errplt = fig[2, :] = Axis(fig; title = "HSIC Objective",
                               xlabel = "Time (sec)",
                               ylabel = "Error")
lines!(errplt, recording.t, recording.lossx; label = "HSIC(X, Z)", color = :green)
lines!(errplt, recording.t, recording.lossy; label = "HSIC(Y, Z)", color = :red)
lines!(errplt, recording.t, recording.lossx .- 10 .* recording.lossy;
       label = "HSIC(X, Z) - 2 HSIC(Y, Z)", color = :blue)
vlines!(errplt, [Tinit]; color = :black, linestyle = :dash)
xlims!(errplt, Tinit - 10, Tinit + nepochs * Ttrain)
fig[3, :] = Legend(fig, errplt; orientation = :horizontal, tellheight = true)

save("output/lif-dense-test.pdf", fig)
