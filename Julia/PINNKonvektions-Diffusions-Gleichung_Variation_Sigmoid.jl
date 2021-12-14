using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cuba, CUDA
using DelimitedFiles
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

inner_array = [16, 32, 64]

act_Fkt = Flux.sigmoid #Flux.relu, Flux.sigmoid, Flux.Gelu, tanh

function create_PINN(eq, bcs, domains, chain, inner, hidden)  
    
    initθ = CuArray(Float64.(DiffEqFlux.initial_params(chain))) #CuArray

    strategy = GridTraining(1/64)

    discretization = PhysicsInformedNN(chain,strategy; init_params = initθ)

    @named pde_system = PDESystem(eq,bcs,domains,[x,y],[ϕ(x, y)])
    prob = discretize(pde_system,discretization)

    Loss = []

    cb = function (p,l)
        #println("Current loss is: $l") 
        push!(Loss, l)
        if l < 1e-5
            return true 
        else 
            return false
        end
    end

    GalacticOptim.solve(prob,ADAM(0.001);cb=cb, maxiters=10000)


    #CSV-Datei erstellen
    name = "/home/pi/stolzean/ErgebnissePINNs/PINN_Sig_nodes$(inner)_hidden$(hidden).csv"
    writedlm(name, Loss, ',') 

end




for inner in inner_array
    println("Sigmoid: Prozess startet mit $(inner)!")

    #2 hidden layer
    chain1 = FastChain(FastDense(2,inner,act_Fkt),
                  FastDense(inner,inner,act_Fkt),
                  FastDense(inner,1))
    time = @elapsed create_PINN(eq, bcs, domains, chain1, inner, 2)
    println("Time 2 HiddenLayer + $(inner) Nodes: ", time)


    #4 hidden layer
    chain2 = FastChain(FastDense(2,inner,act_Fkt),
                  FastDense(inner,inner,act_Fkt),
                  FastDense(inner,inner,act_Fkt),
                  FastDense(inner,inner,act_Fkt),
                  FastDense(inner,1))
    time = @elapsed create_PINN(eq, bcs, domains, chain2, inner, 4)
    println("Time 4 HiddenLayer + $(inner) Nodes: ", time)

    
    #6 hidden layer
    chain3 = FastChain(FastDense(2,inner,act_Fkt),
                  FastDense(inner,inner,act_Fkt),
                  FastDense(inner,inner,act_Fkt),
                  FastDense(inner,inner,act_Fkt),
                  FastDense(inner,inner,act_Fkt),
                  FastDense(inner,inner,act_Fkt),
                  FastDense(inner,1))
    time = @elapsed create_PINN(eq, bcs, domains, chain3, inner, 6)
    println("Time 6 HiddenLayer + $(inner) Nodes: ", time)

end

print("Beendet!")


