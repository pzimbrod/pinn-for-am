using Pkg
Pkg.activate(".")
Pkg.instantiate()
# We implement a calar linear advection equation in 1D with unit velocity
# u_t + u_x = 0
using CUDA

coordinates_min = -1.0
coordinates_max = 1.0

# We assume periodic boundaries and the following IC
# IC(x) = 1.0 + 0.5 * sin(pi*x)
@. IC(x) = 1.0 * ( x >= -0.5 && x <= 0.5)

# We create a discretization with some elements
n_elements = 1024
dx = (coordinates_max - coordinates_min) / n_elements 

using FastGaussQuadrature, Jacobi
polydeg = 3
nodes, weights = gausslobatto(polydeg+1)

# Next, represent the IC in the newly created Galerkin subspace
# But first, we need to do the mapping between the global and local coordinates of the elements
# Our x-Array holds the nodal coordinates of the entire domain
# The rows are the individual element nodes, the columns the elements
x = Matrix{Float64}(undef,length(nodes), n_elements) 

xl = zeros(eltype(nodes),n_elements)
elements = 1:n_elements
xl = (elements .- 1) .* dx .+ dx/2 .+ coordinates_min 
x = xl' .+ dx/2 .* nodes |> CuArray

u0 = IC.(x) |> CuArray

# Generating the variational form
using LinearAlgebra
M = diagm(Float64.(weights)) |> CuArray
# This is our mass matrix. We brought it into a diagonal form so that is is invertible easily.
# Normally with exact integration, this would be a dense matrix. Summing all elements onto the
# Diagonal as a result of numerical integration is called mass lumping

# Now, to deal with the surface term (i.e. the numerical flux), we create a boundary matrix
B = diagm([-1;zeros(polydeg-1);1]) |> CuArray
# This is a simple upwind Flux.

# The following functions are shamelessly stolen from Trixi.jl
function barycentric_weights(nodes)
  n_nodes = length(nodes)
  weights = ones(n_nodes)

  for j = 2:n_nodes, k = 1:(j-1)
    weights[k] *= nodes[k] - nodes[j]
    weights[j] *= nodes[j] - nodes[k]
  end

  for j in 1:n_nodes
    weights[j] = 1 / weights[j]
  end

  return weights
end

function polynomial_derivative_matrix(nodes)
  n_nodes = length(nodes)
  d = zeros(n_nodes, n_nodes)
  wbary = barycentric_weights(nodes)

  for i in 1:n_nodes, j in 1:n_nodes
    if j != i
      d[i, j] = wbary[j] / wbary[i] * 1 / (nodes[i] - nodes[j])
      d[i, i] -= d[i, j]
    end
  end

  return d
end

# Derivative matrix of our Legendre-Gauss-Lobatto Basis
D = polynomial_derivative_matrix(nodes) |> CuArray

function rhs!(du,u,x,t)
    du .= zero(eltype(du))
    flux_numerical = copy(du)

    # GPU views to avoid scalar indexing
    flux_ll = view(flux_numerical, 1, 2:size(du,2)-1)
    flux_lr = view(flux_numerical, size(du,1), 1:size(du,2)-2)

    flux_rl = view(flux_numerical, size(du,1), 2:size(du,2)-1)
    flux_rr = view(flux_numerical, 1, 3:size(du,2))

    u_ll = view(u,1,2:size(du,2)-1)
    u_lr = view(u, size(du,1), 1:size(du,2)-2)
    u_rl = view(u, size(du,1), 2:size(du,2)-1)
    u_rr = view(u, 1, 3:size(du,2))

    @. flux_ll = -0.5 * 1.0 * (u_ll - u_lr)
    @. flux_lr = flux_ll
    @. flux_rl = -0.5 * 1.0 * (u_rl - u_rr)
    @. flux_rr = flux_rl

    #= for i ∈ 2:size(du,2)-1
        # left interface
        @views flux_numerical[1,i] = -0.5 * 1.0 * (u[1,i] - u[end,i-1])
        @views flux_numerical[end,i-1] = flux_numerical[1,i]

        # right interface
        @views flux_numerical[end,i] = -0.5 * 1.0 * (u[1,i+1] - u[end,i])
        @views flux_numerical[1,i+1] = flux_numerical[end,i]
    end =#
    # Boundary flux
    flux_left = view(flux_numerical, 1,1)
    flux_right = view(flux_numerical, size(flux_numerical)...)
    u_left = view(u,1,1)
    u_right = view(u,size(u)...)
    @. flux_left = -0.5 * 1.0 * (u_left - u_right)
    @. flux_right = flux_left
    #= @views flux_numerical[1,1] = -0.5 * 1.0 * (u[1,1] - u[end,end])
    @views flux_numerical[end,end] = flux_numerical[1,1] =#
    
    # Calculate surface integrals
    du -= (M \ B) * flux_numerical

    # Calculate volume integral
    du += (M \ D') * M * u

    # Apply Jacobian from mapping to reference element
    du .*= 2 / dx

    return nothing
end

using OrdinaryDiffEq

tspan = (0.0, 2.0) 
ode = ODEProblem(rhs!,u0,tspan,x)
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6, save_everystep=false)

using BenchmarkTools
t = @benchmark CUDA.@sync solve($ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6, save_everystep=false)
print(dump(t))
