using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cuba, CUDA
import ModelingToolkit: Interval

@parameters x y t
@variables ϕ(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dx = Differential(x)
Dt = Differential(t)

x_min = 0.
x_max = 1.
y_min = 0.
y_max = 1.
t_min = 0.
t_max = 1.
κ = 0.1
f = 0
u(x) = 1.4 * x - 0.4


# 2D PDE
eq  = Dt(ϕ(x,y,t)) + u(x) * Dx(ϕ(x,y,t)) - κ * Dxx(ϕ(x,y,t)) - κ * Dyy(ϕ(x,y,t)) ~ f


# Space and time domains
domains = [x ∈ Interval(x_min,x_max),
           y ∈ Interval(y_min,y_max),
	   t ∈ Interval(t_min,t_max)]


analytic_sol_func(y,t) = sin(5 * π * y) * exp(-t)
# Initial and boundary conditions
bcs = [ϕ(x, y_min, t) ~ analytic_sol_func(y_min, t),
        ϕ(x, y_max, t) ~ analytic_sol_func(y_max, t),
	ϕ(x,y,t_min) ~ analytic_sol_func(y, t_min)]


# Neural network
inner = 32
chain = FastChain(FastDense(3,inner,Flux.relu),
                  FastDense(inner,inner,Flux.relu),
                  FastDense(inner,inner,Flux.relu),
                  FastDense(inner,inner,Flux.relu),
                  FastDense(inner,1))


initθ = CuArray(Float64.(DiffEqFlux.initial_params(chain)))

strategy = GridTraining(1/64)

discretization = PhysicsInformedNN(chain,strategy; init_params = initθ)

@named pde_system = PDESystem(eq,bcs,domains,[x,y,t],[ϕ(x, y, t)])
prob = discretize(pde_system,discretization)

#MSE Loss Function
cb = function (p,l)
    println("Current loss is: $l") 
    return false
end

res = GalacticOptim.solve(prob,ADAM(0.001);cb=cb,reltol=1e-5,maxiters=10000)
