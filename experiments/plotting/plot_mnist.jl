using CairoMakie
using DataFrames, CSV
using Statistics: mean, std

bp_path = "output/bp_data"
hsic_path = "output/mnist_data"

##

ntrials = 1
nepochs = 100
validation_points = [1, 5, ntuple(i -> i * 10, nepochs ÷ 10)...]
accuracies = Dict{String, Matrix{Float32}}(["Back-propagation" => zeros(Float32, length(validation_points), ntrials),
                                            "Our method" => zeros(Float32, length(validation_points), ntrials)])

##

for trial in 1:ntrials
    df = CSV.read(joinpath(bp_path, "trial_$trial.csv"), DataFrame)
    accuracy = skipmissing(df[:, :accuracies]) |> collect
    accuracies["Back-propagation"][:, trial] .= accuracy .* 100

    df = CSV.read(joinpath(hsic_path, "trial_$trial.csv"), DataFrame)
    accuracy = skipmissing(df[:, :accuracies]) |> collect
    accuracies["Our method"][:, trial] = accuracy .* 100
end

##

fig = Figure()

colors = (:blue, :orange)
ax = fig[1, 1] = Axis(fig; title = "Average MNIST (Subset) Test Accuracy (4 Trials)",
                           xlabel = "Epochs",
                           ylabel = "Test Accuracy (%)")
avgplts = Dict{String, Any}()
sctplts = Dict{String, Any}()
errplts = Dict{String, Any}()
for ((method, accuracy), c) in zip(pairs(accuracies), colors)
    μ = dropdims(mean(accuracy; dims = 2); dims = 2)
    σ = dropdims(std(accuracy; dims = 2); dims = 2)
    avgplts[method] = lines!(ax, validation_points, μ; color = c)
    sctplts[method] = scatter!(ax, validation_points, μ; color = c)
    errplts[method] = errorbars!(ax, validation_points, μ, σ; whiskerwidth = 10, color = :red)
end
Legend(fig[2, 1], [[avgplts[m], sctplts[m]] for m in keys(accuracies)], collect(keys(accuracies));
       orientation = :horizontal, tellheight = true)

CairoMakie.save("output/mnist-subset-accuracy.pdf", fig)
