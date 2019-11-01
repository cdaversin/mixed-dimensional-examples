from utils import *

n_values = [16, 32, 64, 128]
order = 1

def stokes_brinkman(n):

    # Mesh
    mesh = UnitSquareMesh(n, n)

    gamma = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    for f in facets(mesh):
        if 1.0 - DOLFIN_EPS < f.midpoint().x() < 1.0 + DOLFIN_EPS:
            gamma[f] = 3
        elif 1.0 - DOLFIN_EPS < f.midpoint().y() < 1.0 + DOLFIN_EPS:
            gamma[f] = 0
        elif - DOLFIN_EPS < f.midpoint().x() <  DOLFIN_EPS:
            gamma[f] = 1
        elif - DOLFIN_EPS < f.midpoint().y() <  DOLFIN_EPS:
            gamma[f] = 2

    submesh = MeshView.create(gamma, 1)

    # P(k+1)-P(k)-P(k)
    V = VectorFunctionSpace(mesh, 'CG', order + 1)
    Q = FunctionSpace(mesh, 'CG', order)
    M = VectorFunctionSpace(submesh, 'CG', order)

    W = MixedFunctionSpace(V, Q, M)

    (u, p, lambda_) = TrialFunctions(W)
    (v, q, beta_) = TestFunctions(W)

    dxOmega = Measure("dx", domain=mesh)
    dxGamma = Measure('dx', domain=submesh)

    # Variational formulation
    a00 = inner(grad(u), grad(v))*dxOmega + inner(u, v)*dxOmega
    a01 = inner(p, div(v))*dxOmega
    a02 = inner(v, lambda_)*dxGamma
    a10 = inner(q, div(u))*dxOmega
    a20 = inner(u, beta_)*dxGamma

    a = a00 + a01 + a02 + a10 + a20
    
    u_true, rhs_data = setup_mms(1.)
    (f,h,u0) = rhs_data
    
    L0 = inner(f, v)*dxOmega + inner(h, v)*ds(domain=mesh, subdomain_data=gamma, subdomain_id=3)
    L2 = inner(u0, beta_)*dxGamma
    L = L0 + L2
    
    sol = Function(W)

    system = assemble_mixed_system(a == L, sol)
    matrix_blocks = system[0]
    rhs_blocks = system[1]

    # Assembly and Solve
    solve(a == L, sol, solver_parameters={"linear_solver":"direct"})
    return [sol.sub(0), sol.sub(1), sol.sub(2)]

## Main function
nprocs = MPI.size(MPI.comm_world) 
v_data = open("stokes-brinkman-v_P" + str(order+1) + "_np"+str(nprocs) + "_cvg.dat","w+")
v_data.write( "N\tL2\tH1\n" )
p_data = open("stokes-brinkman-p_P" + str(order) + "_np"+str(nprocs) + "_cvg.dat","w+")
p_data.write( "N\tL2\tH1\n" )

for n in n_values:
    print('n = ', n)

    # Manufactured solution
    u_true, rhs_data = setup_mms(1.)
    approx = stokes_brinkman(n)
    exact = list(u_true)

    # Write errors to file
    err_v_L2 = errornorm(exact[0], approx[0], 'L2')
    err_v_H1 = errornorm(exact[0], approx[0], 'H1')
    print('v - L2 error = ', err_v_L2)
    print('v - H1 error = ', err_v_H1)
    v_data.write( str(n) + "\t" + str(err_v_L2) + "\t" + str(err_v_H1) + "\n" )

    err_p_L2 = errornorm(exact[1], approx[1], 'L2')
    err_p_H1 = errornorm(exact[1], approx[1], 'H1')
    print('p - L2 error = ', err_p_L2)
    print('p - H1 error = ', err_p_H1)
    p_data.write( str(n) + "\t" + str(err_p_L2) + "\t" + str(err_p_H1) + "\n" )

## Export approx
out_u = File("stokes-brinkman-v.pvd")
out_p = File("stokes-brinkman-p.pvd")
out_l = File("stokes-brinkman-l.pvd")
out_u << approx[0]
out_p << approx[1]
out_l << approx[2]
