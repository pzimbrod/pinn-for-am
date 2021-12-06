using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cuba, CUDA
import ModelingToolkit: Interval

@parameters x y
@variables ϕ(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dx = Differential(x)

x_min = 0.
x_max = 1.
y_min = 0.
y_max = 1.
κ = 0.1
f = 0
u(x) = 1.4 * x - 0.4


# 2D PDE
eq  = u(x) * Dx(ϕ(x,y)) - κ * Dxx(ϕ(x,y)) - κ * Dyy(ϕ(x,y)) ~ f


# Space and time domains
domains = [x ∈ Interval(x_min,x_max),
           y ∈ Interval(y_min,y_max)]


analytic_sol_func(y) = sin(5 * π * y)
# Initial and boundary conditions
bcs = [ϕ(x, y_min) ~ analytic_sol_func(y_min),
        ϕ(x, y_max) ~ analytic_sol_func(y_max)]


# Neural network
inner = 32
chain = FastChain(FastDense(2,inner,Flux.relu),
                  FastDense(inner,inner,Flux.relu),
                  FastDense(inner,inner,Flux.relu),
                  FastDense(inner,inner,Flux.relu),
                  FastDense(inner,1))

#### Welches Verfahren?
#### CuArray(Float64 ....)
initθ = Float64.(DiffEqFlux.initial_params(chain))

strategy = GridTraining(1/64)

discretization = PhysicsInformedNN(chain,strategy; init_params = initθ)

@named pde_system = PDESystem(eq,bcs,domains,[x,y],[ϕ(x, y)])
prob = discretize(pde_system,discretization)

#MSE Loss Function
cb = function (p,l)
    println("Current loss is: $l") 
    return false
end

res = GalacticOptim.solve(prob,ADAM(0.001);cb=cb,maxiters=2)
