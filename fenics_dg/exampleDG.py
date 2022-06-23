from dolfin import *

# Length of domain in x and y direction
Lx = 5.0
Ly = 5.0
# Number of elements in x and y direction
Nx = 50
Ny = 50

# Degree of polynomial approximation
polydeg = 5

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
u = interpolate( Constant(("1.", "1")), V_u)

# The test function of the weak form
v = TestFunction(V_dg)
# The interpolation function for the scalar quantity
phi = TrialFunction(V_dg)

kappa = Constant(0.0)
f = Constant(0.0)
alpha = Constant(5.0)

n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

un = (dot(u,n) + abs(dot(u,n)))/2.0

a_int = dot(grad(v), kappa*grad(phi) - u * phi) * dx

a_fac = kappa('+') * (alpha('+') / h('+')) * dot(jump(v,n), jump(phi,n)) * dS \
- kappa('+') * dot(avg(grad(v)), jump(phi, n)) * dS \
- kappa('+') * dot(jump(v,n), avg(grad(phi))) * dS

a_vel = dot(jump(v), un('+') * phi('+') - un('-') * phi('-') ) * dS \
+ dot(v, un*phi) * ds

a = a_int + a_fac + a_vel

L = v * f * dx

phi_h = Function(V_dg)

A = assemble(a)
b = assemble(L)
# Periodic BCs get imposed directly, hence don't need to included into FE assembly
# bc.apply(A, b)

solve(A, phi_h.vector(), b)

# Project solution to a continuous function space
up = project(phi_h, V=V_cg)

file = File("exampleDG/scalar_transport.pvd")
file << up
