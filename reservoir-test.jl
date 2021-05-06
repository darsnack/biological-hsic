using DifferentialEquations
using GLMakie

##

function lorenz!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz!,u0,tspan)
sol = solve(prob)

##

fig = Figure()
fig[1, 1] = Axis(fig; title = "Solution to the linear ODE with a thick line",
                      xlabel = "Time (t)",
                      ylabel = "u(t) (in μm)")
lines!(fig[1, 1], sol.t, sol.u; label = "My Thick Line!")
lines!(fig[1, 1], sol.t, map(t -> 0.5 * exp(1.01 * t), sol.t);
       linewidth = 3, linestyle = :dash,
       label = "True Solution!")
display(current_figure())