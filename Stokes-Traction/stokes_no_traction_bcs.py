from dolfin import *

##### ---------- Parameters ----------- #####
alpha = 1.0
# mu : Dynamic viscosity
# water : 1e-3 kg.m-1.s-1
mu = 1e-3
# rho : fluid density
# water : 1000 kg.m-3
rho = 1e3
# p0_in/p0_out : Imposed pressures at inlet/outlet 
p0_in = 8
p0_out = 10
f = Constant(("1e-3","0.0"))
##### -------------------------------- #####

##### ---------- Meshes - Function Spaces ----------- #####
n1 = 80
n2 = 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(10.0, 5.0), n1, n2)
physical_facets = MeshFunction("size_t", mesh, 1, 0)

CompiledSubDomain('near(x[0], 0)').mark(physical_facets, 101)
CompiledSubDomain('near(x[0], 10.0)').mark(physical_facets, 102)
CompiledSubDomain('near(x[1]*(x[1]-5.0), 0)').mark(physical_facets, 100)

V = VectorFunctionSpace(mesh, "CG", 2)
M = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace(V,M)

# Trial (u,p)
(u,p) = TrialFunctions(W)
# Test (v,q)
(v,q) = TestFunctions(W)

dV = Measure("dx", domain=W.sub_space(0).mesh())
dV_boundary = Measure("ds", domain=W.sub_space(0).mesh(), subdomain_data=physical_facets)

##### ----------------------------------------------- #####

##### -------------- Variational form --------------- #####

n = FacetNormal(mesh)
t_lm = Expression(("0.0","-1.0"), degree=1)

defu = sym(grad(u))
defv = sym(grad(v))

a0 = 2*mu*inner(defu,defv)*dV - div(v)*p*dV - div(u)*q*dV 
apq = Constant(0.0)*inner(p,q)*dV
## Bilinear form
a = a0 + apq
## Linear form
L = rho*inner(f,v)*dV - p0_in*inner(v,n)*dV_boundary(101) - p0_out*inner(v,n)*dV_boundary(102)

##### ----------------------------------------------- #####

##### ----------- Boundary conditions --------------- #####

# Boundary conditions on gamma1 (wall)
zero = Constant(("0.0","0.0"))
bc_wall = DirichletBC(W.sub_space(0), zero, physical_facets, 100)
bcs = [bc_wall]

##### ----------------- Solve ----------------------- #####

sol = Function(W)
solve(a == L, sol, bcs, solver_parameters={"linear_solver":"direct"})

out_u = File("Stokes-NoTractionBCs-v.pvd")
out_p = File("Stokes-NoTractionBCs-p.pvd")
out_u << sol.sub(0)
out_p << sol.sub(1)

