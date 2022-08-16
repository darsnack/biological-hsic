function reservoir_test(data, Nhidden, target;
                        η0 = 1f-4,
                        τ = 50f-3, # LIF time constant
                        λ = 1.7, # chaotic level
                        τavg = 5f-3, # signal smoothing constant
                        Tinit = 50f0, # warmup time
                        Ttrain = 500f0, # training time
                        Ttest = 100f0, # testing time
                        Δt = 1f-3, # simulation time step
                        Δtsample = 50f-3, # time to present each data sample
                        bs = 6) # effective batch size
    
end