from dolfin import *

n_values = [2, 4, 8, 16, 32]
exact = Expression("x[0]*(1-x[0])", degree=2)
order= 1

def poisson(n):
    mesh = UnitCubeMesh(n, n, n)
    
    marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    for f in facets(mesh):
        marker[f] = 0.5 - DOLFIN_EPS < f.midpoint().x() < 0.5 + DOLFIN_EPS

    ## Create submesh ##
    submesh = MeshView.create(marker, 1)

    ## Setup formulation ##
    V = FunctionSpace(mesh, "CG", order)
    LM = FunctionSpace(submesh, "CG", order)
    W = MixedFunctionSpace(V,LM)

    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
    bc = DirichletBC(V, Constant(0.0), boundary)

    (u,l) = TrialFunctions(W)
    (v,e) = TestFunctions(W)

    dV = Measure("dx", domain=W.sub_space(0).mesh())
    dL = Measure("dx", domain=W.sub_space(1).mesh())

    fu = Constant(2.)
    fl = Constant(0.25)

    a = inner(grad(u),grad(v))*dV + v*l*dL + u*e*dL
    L = fu*v*dV + fl*e*dL

    sol = Function(W)
    ## Assembly and Solve ##
    solve(a == L, sol, bc, solver_parameters={"linear_solver":"direct"})
    return (sol.sub(0), sol.sub(1))

## Main function
nprocs = MPI.size(MPI.comm_world) 
u_data = open("poisson_3D_P"+str(order)+"_np"+str(nprocs)+"_cvg.dat","w+")
u_data.write( "N\tL2\tH1\n" )

for n in n_values:
    print('n = ', n)

    (approx_u, approx_l) = poisson(n)
    err_u_L2 = errornorm(exact, approx_u, 'L2')
    err_u_H1 = errornorm(exact, approx_u, 'H1')

    # Write errors to file
    print('L2 error = ', err_u_L2)
    print('H1 error = ', err_u_H1)
    u_data.write( str(n) + "\t" + str(err_u_L2) + "\t" + str(err_u_H1) + "\n" )

## Export last solution
out_sub0 = File("poisson-lm-3D-0.pvd")
out_sub1 = File("poisson-lm-3D-1.pvd")
out_sub0 << approx_u
out_sub1 << approx_l
