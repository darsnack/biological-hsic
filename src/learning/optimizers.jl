struct WeightDecayChain{T, S}
    opts::T
end

WeightDecayChain(opt, decay) = WeightDecayChain(Flux.Optimiser(WeightDecay(decay), opt))

function Base.getproperty(opt::WeightDecayChain, name)
    if name == :wd
        return opt.opts[1].wd
    elseif name == :opts
        return opt.opts
    else
        return getproperty(opt.opts[2], name)
    end
end

function Base.setproperty!(opt, name, value)
    if name == :wd
        return setproperty!(opt.opts[1], :wd, value)
    elseif name == :opts
        throw(ArgumentError("Cannot set property opt.opts."))
    else
        return setproperty!(opt.opts[2], name, value)
    end
end

Flux.Optimise.apply!(opt::WeightDecayChain, x, dx) = Flux.Optimise.apply!(opt.opts, x, dx)
