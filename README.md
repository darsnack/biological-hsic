# Information Bottleneck-Based Hebbian Learning Rule Naturally Ties Working Memory and Synaptic Updates

A biologically plausible implementation of the HSIC objective for training SNNs.

## Setup instructions

To run the code, you need to install Julia 1.6. This can be obtained by going to https://julialang.org/downloads/ and following the installation instructions.

Once you install Julia, you need to install all the packages used by the code. This can be done by launching Julia (i.e. `julia`) in the repository root directory. Next, activate the project environment and instantiate the packages. Do to this, enter [Pkg mode](https://docs.julialang.org/en/v1.6/stdlib/REPL/#Pkg-mode) by pressing `]`. Then run the following at the Julia REPL.

```julia
(@1.6)> activate .
(biological-hsic)> instantiate
```

## Running the experiments

The code for running the experiments is provided as Julia scripts under the `experiments/` directory. The directory is organized as follows:
- `experiments/small`: Small-scale experiments (i.e. synthetic datasets w/ the reservoir simulated)
- `experiments/mnist`: Large-scale MNIST experiments (i.e. reservoir is not directly simulated)
- `experiments/plotting`: Plotting scripts for generating the large-scale experiment figures

### Small-scale experiments

To run the reservoir experiment, launch a Julia REPL, activate the project environment, and run:
```julia
julia> include("experiments/small/reservoir-hsic.jl")
```
The resulting figure will be under `output/reservoir-hsic.pdf`.

To run the linear synthetic dataset experiment, launch a Julia REPL, activate the project environment, and run:
```julia
julia> include("experiments/small/linear.jl")
```
The resulting figure will be under `output/linear.pdf`.

To run the `tanh` synthetic dataset experiment, launch a Julia REPL, activate the project environment, and run:
```julia
julia> include("experiments/small/tanh.jl")
```
The resulting figure will be under `output/tanh.pdf`.

### Large-scale experiments

Most of the large-scale experiments generate data stored as CSV files under `output/`. To create the figures in the paper, you need to run the scripts under `experiments/plotting/`.

#### Generate the data

To generate the MNIST back-propagation data, launch a Julia REPL, activate the project environment, and run:
```julia
julia> include("experiments/mnist/mnist-bp.jl")
```

To generate the MNIST pHSIC (prior work) data, launch a Julia REPL, activate the project environment, and run:
```julia
julia> include("experiments/mnist/mnist-phsic.jl")
```

To generate the MNIST data for our method, launch a Julia REPL, activate the project environment, and run:
```julia
julia> include("experiments/mnist/mnist.jl")
```

#### Plot the results

To plot the results comparing our method and back-propagation over epochs, launch a Julia REPL, activate the project environment, and run:
```julia
julia> include("experiments/plotting/plot_mnist.jl")
```

To plot the results comparing our method, back-propagation, and pHSIC (prior work), launch a Julia REPL, activate the project environment, and run:
```julia
julia> include("experiments/plotting/plot_baselines.jl")
```
