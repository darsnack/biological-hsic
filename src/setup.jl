using Adapt
using LoopVectorization
using CUDA
CUDA.allowscalar(false)
# using FLoops, FoldsCUDA

# using ProgressLogging
# using Logging: global_logger, with_logger
# using TerminalLoggers: TerminalLogger
# global_logger(TerminalLogger())

using BangBang
using CircularArrays
using CircularArrayBuffers
using Distributions, Random
using Distances
using DataStructures
using Flux
using Flux: Zygote, @functor, Recur
using FluxTraining
using FluxTraining: AbstractTrainingPhase
using LinearAlgebra
using MLUtils, MLDatasets
using NNlib
using ParameterSchedulers
using TensorCast
using TensorCast: @reduce
using Wandb

include("utils.jl")
include("hsic.jl")
include("networks/reservoir.jl")
# include("networks/dense.jl")
# include("networks/conv.jl")
# include("networks/chain.jl")
include("networks/utils.jl")
include("data.jl")
include("learning/utils.jl")
include("learning/rmhebb.jl")
include("learning/globalerror.jl")
# include("learning/learners.jl")
include("learning/optimizers.jl")
include("loops.jl")
