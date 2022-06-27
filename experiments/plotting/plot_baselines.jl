using CairoMakie
using DataFrames, CSV

bp_path = "output/bp_data"
phsic_path = "output/phsic_data"
hsic_path = "output/mnist_data"

##

methods = ["Back-propagation", "pHSIC (bs = 256)", "pHSIC (bs = 2)", "Our method (bs = 32)"]
accuracies = Dict{String, Float32}(["Back-propagation" => 0,
                                    "pHSIC (bs = 256)" => 94.6, # taken from paper
                                    "pHSIC (bs = 2)" => 0,
                                    "Our method (bs = 32)" => 0])

##

df = CSV.read(joinpath(bp_path, "trial_1.csv"), DataFrame)
accuracy = skipmissing(df[:, :accuracies]) |> collect |> last
accuracies["Back-propagation"] = accuracy * 100

df = CSV.read(joinpath(phsic_path, "trial_1.csv"), DataFrame)
accuracy = skipmissing(df[:, :accuracies]) |> collect |> last
accuracies["pHSIC (bs = 2)"] = accuracy * 100

df = CSV.read(joinpath(hsic_path, "trial_1.csv"), DataFrame)
accuracy = skipmissing(df[:, :accuracies]) |> collect |> last
accuracies["Our method (bs = 32)"] = accuracy * 100

##

fig = Figure()

ax = fig[1, 1] = Axis(fig; title = "MNIST Test Accuracy",
                           xlabel = "Methods",
                           ylabel = "Test Accuracy (%)")
bplt = barplot!(ax, 1:length(methods), [accuracies[m] for m in methods])
ax.xticks = (1:length(methods), methods)

CairoMakie.save("output/baseline-accuracies.pdf", fig)
