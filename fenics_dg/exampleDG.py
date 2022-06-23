from dolfin import *

# Don't receive messages from every subprocess
parameters["std_out_all_processes"] = False;

# In order to make MPI work, facet information needs to be exchanged
parameters["ghost_mode"] = "shared_facet" 

# Length of domain in x and y direction
Lx = 5.0
Ly = 5.0
# Number of elements in x and y direction
Nx = 50
Ny = 50

# transport velocity
vel_x = 1.0
vel_y = 1.0
vel = sqrt(vel_x+vel_y)
# grid spacing
h = min(Lx/Nx, Ly/Ny)
# Necessary max time step such that CFL = 0.8 (to be conservative)
dt = 0.8 * h/vel
# End time
t_end = 5.0

# Degree of polynomial approximation
polydeg = 1

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], 1)) or 
                        (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - Lx
            y[1] = x[1] - Ly
        elif near(x[0], 1):
            y[0] = x[0] - Lx
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - Ly

periodic_boundary = PeriodicBoundary()

#mesh = UnitSquareMesh(64,64)
mesh = RectangleMesh.create([Point(0,0),Point(Lx,Ly)],[Nx,Ny],CellType.Type.quadrilateral)

# Discontinuous function space for the scalar quantity
V_dg = FunctionSpace(mesh, "DG", polydeg, constrained_domain=periodic_boundary)

# Continuous function space for postprocessing
V_cg = FunctionSpace(mesh, "CG", polydeg)

# Advection velocity is a vector, hence a separate vector space is needed
# dim = 2
V_u = VectorFunctionSpace(mesh, "CG", 2)

# project constant, uniform velocity onto function space for u
u = interpolate( Constant((vel_x, vel_y)), V_u)

# Initial condition: alpha = 1 if 2 < (x,y) < 4
ic = Expression('(2 <= x[0]) * (x[0] <= 4) * (2 <= x[1]) * (x[1] <= 4)',
                element=V_dg.ufl_element())
phi_0 = interpolate(ic, V_dg)

# The test function of the weak form
v = TestFunction(V_dg)
# The interpolation function for the scalar quantity
phi = TrialFunction(V_dg)

f = Constant(0.0)

n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

un = (dot(u,n) + abs(dot(u,n)))/2.0

a_int = dot(grad(v), - u * phi) * dx

a_vel = dot(jump(v), un('+') * phi('+') - un('-') * phi('-') ) * dS \
+ dot(v, un*phi) * ds

a = dt * a_int + dt * a_vel

L = phi_0 * v * dx

phi_h = Function(V_dg)

A = assemble(a)
b = assemble(L)
# Periodic BCs get imposed directly, hence don't need to included into FE assembly
# bc.apply(A, b)

# solve(A, phi_h.vector(), b)

# Generate the output file
file = File("fenics_dg/exampleDG/scalar_transport.pvd")

# Time loop
t = 0.0
while t <= t_end:
    
    up = phi_h
    file << up

    # Update time
    t += dt
    print("time step: "+str(t))

    # linear solve
    solve(A, phi_h.vector(), b)
    
    # Update previous solution
    phi_0.assign(phi_h)
