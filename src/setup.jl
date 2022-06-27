using Adapt
using LoopVectorization
using CUDA
CUDA.allowscalar(false)
# using FLoops, FoldsCUDA

using ProgressLogging
using Logging: global_logger, with_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using Distributions, Random
using Distances
using DataStructures
using LinearAlgebra
using TensorCast
using TensorCast: @reduce
using Flux
using Flux: Zygote
using MLUtils, MLDatasets
using CircularArrayBuffers
using NNlib
using BangBang

include("utils.jl")
include("hsic.jl")
include("networks/reservoir.jl")
include("networks/dense.jl")
include("networks/conv.jl")
include("networks/chain.jl")
include("networks/utils.jl")
include("data.jl")
include("learning/utils.jl")
include("learning/globalerror.jl")
include("learning/learners.jl")
include("learning/optimizers.jl")
include("loops.jl")
