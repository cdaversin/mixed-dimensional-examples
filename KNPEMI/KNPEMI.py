from dolfin import *
import numpy as np

from KNPEMI_MMS import MMS

# Export results
do_export = True
# Polynomial order (= 1 or 2)
order = 1

def setup_square_domain(n):
    """ Inner (interior) is [0.25, 0.75]^2, outer (exterior) is
    [0, 1]^2 \ [0.25, 0.75]^2 and \partial [0.25, 0.75]^2 is the interface """

    # square mesh
    mesh = UnitSquareMesh(n, n)
    # define interior domain
    interior = CompiledSubDomain('std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25')
    # create mesh function
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

    # mark interior and exterior domain
    for cell in cells(mesh):
        x = cell.midpoint().array()
        subdomains[cell] = int(interior.inside(x, False))
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    # create exterior mesh
    exterior_mesh = MeshView.create(subdomains, 0)
    # create interior mesh
    interior_mesh = MeshView.create(subdomains, 1)

    # create interface mesh
    surfaces = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    for f in facets(mesh):
        interface_left = (0.25 - DOLFIN_EPS < f.midpoint().x() < 0.25 + DOLFIN_EPS) and (0.25 - DOLFIN_EPS < f.midpoint().y() < 0.75 + DOLFIN_EPS)
        interface_right = (0.75 - DOLFIN_EPS < f.midpoint().x() < 0.75 + DOLFIN_EPS) and (0.25 - DOLFIN_EPS < f.midpoint().y() < 0.75 + DOLFIN_EPS)
        interface_bottom = (0.25 - DOLFIN_EPS < f.midpoint().y() < 0.25 + DOLFIN_EPS) and (0.25 - DOLFIN_EPS < f.midpoint().x() < 0.75 + DOLFIN_EPS)
        interface_top = (0.75 - DOLFIN_EPS < f.midpoint().y() < 0.75 + DOLFIN_EPS) and (0.25 - DOLFIN_EPS < f.midpoint().x() < 0.75 + DOLFIN_EPS)
        surfaces[f] = interface_left or interface_right or interface_bottom or interface_top
    gamma_mesh = MeshView.create(surfaces, 1)
    
    # Mark the outside of exterior mesh
    facet_f = MeshFunction('size_t', exterior_mesh, exterior_mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0]*(1-x[0]), 0) || near(x[1]*(1-x[1]), 0)').mark(facet_f, 1)
    exterior_mesh.subdomains = facet_f

    return interior_mesh, exterior_mesh, gamma_mesh

def solve_system(n, t, dt, Tstop, ion_list, M, order):

    """ Solve the system """
    # get mesh
    interior_mesh, exterior_mesh, gamma_mesh = setup_square_domain(n)

    dt_inv = Constant(1./dt)    # invert time step
    dt = Constant(dt)           # make dt Constant
    # get physical parameters
    psi = 1.0
    F = 1.0
    tmp = 1.0
    R = 1.0
    C_M = 1.0

    N_ions = len(ion_list)

    # get MMS terms and exact solutions
    src_terms, exact_sols, init_conds, bndry_terms, \
                subdomains_MMS =  M.get_MMS_terms_KNPEMI(t)

    # ------------------------- Setup function spaces ----------------------- #
    # Element over interior mesh
    P1 = FiniteElement('P', interior_mesh.ufl_cell(), 1)    # ion concentrations and potentials
    P2 = FiniteElement('P', interior_mesh.ufl_cell(), 2)
    P3 = FiniteElement('P', interior_mesh.ufl_cell(), 3)
    Pk = [P1,P2,P3]
    R0 = FiniteElement('R', interior_mesh.ufl_cell(), 0)    # Lagrange to enforce /int phi_i = 0
    # Element over gamma mesh
    Q1 = FiniteElement('P', gamma_mesh.ufl_cell(), 1)        # membrane ion channels
    Q2 = FiniteElement('P', gamma_mesh.ufl_cell(), 2)
    Q3 = FiniteElement('P', gamma_mesh.ufl_cell(), 3)
    Qk = [Q1,Q2,Q3]
    
    # Intracellular ion concentrations for each ion (N_ion), potential, Lagrange multiplier
    interior_element_list = [Pk[order-1]]*(N_ions + 1) + [R0]
    # Extracellular ion concentrations for each (N_ion), potential
    exterior_element_list = [Pk[order-1]]*(N_ions + 1)

    # Create function spaces
    Wi = FunctionSpace(interior_mesh, MixedElement(interior_element_list))
    We = FunctionSpace(exterior_mesh, MixedElement(exterior_element_list))
    Wg = FunctionSpace(gamma_mesh, Qk[order-1])
    W = MixedFunctionSpace(Wi, We, Wg)
    
    # mark exterior subdomain - subdomains_MMS[0] = (x=0) U (x=1) U (y=0) U (y=1)
    exterior_subdomains = subdomains_MMS[0]
    exterior_boundary = MeshFunction('size_t', exterior_mesh, exterior_mesh.topology().dim()-1, 0)
    [subd.mark(exterior_boundary, 1) for subd in map(CompiledSubDomain, exterior_subdomains)]
    # normal on exterior boundary
    n_outer = FacetNormal(exterior_mesh)
    # measure on exterior boundary
    dsOuter = Measure('ds', domain=exterior_mesh, subdomain_data=exterior_boundary, subdomain_id=1)
    
    # mark interface - subdomains_MMS[1] = (x=0.25) U (x=0.75) U (y=0.25) U (y=0.75)
    gamma_subdomains = subdomains_MMS[1]
    # Mark interface
    gamma_boundary = MeshFunction('size_t', gamma_mesh,gamma_mesh.topology().dim(), 0)
    [subd.mark(gamma_boundary, i) for i, subd in enumerate(map(CompiledSubDomain, gamma_subdomains), 1)]  
    # measures on exterior mesh
    dxe = Measure('dx', domain=exterior_mesh)    
    # measure on interior mesh
    dxi = Measure('dx', domain=interior_mesh)
    # measure on gamma
    dxGamma = Measure('dx', domain=gamma_mesh, subdomain_data=gamma_boundary)

    # ------------------------- Setup functions ----------------------------- #
    # create functions
    (ui, ue, p_IM) = TrialFunctions(W)
    (vi, ve, q_IM) = TestFunctions(W)
    u_p = Function(W)
    ui_p = u_p.sub(0)
    ue_p = u_p.sub(1)

    # split unknowns
    ui = split(ui)
    ue = split(ue)
    # split test functions
    vi = split(vi)
    ve = split(ve)
    # split previous solution
    ui_prev = split(ui_p)
    ue_prev = split(ue_p)

    # intracellular potential
    phi_i = ui[N_ions]              # unknown
    vphi_i = vi[N_ions]             # test function
    # extracellular potential
    phi_e = ue[N_ions]              # unknown
    vphi_e = ve[N_ions]             # test function
    # Lagrange multiplier for /int phi_i = 0
    _c = ui[N_ions+1]               # unknown
    _d = vi[N_ions+1]               # test function

    # get MMS terms and exact solutions
    src_terms, exact_sols, init_conds, bndry_terms, \
                subdomains_MMS =  M.get_MMS_terms_KNPEMI(t)
    # set initial membrane potential
    phi_M_init = init_conds['phi_M']
    phi_M_prev = interpolate(phi_M_init, Wg)
    
    # --------------------- Setup variational formulation ---------------------- #
    # sum of fractions
    alpha_i_sum = 0  # intracellular
    alpha_e_sum = 0  # extracellular
    I_ch = 0         # total channel current

    # Initialize parts of variational formulation
    for idx, ion in enumerate(ion_list):
        # get ion attributes
        z = ion['z']; Di = ion['Di']; De = ion['De'];
        
        # set initial value of intra and extracellular ion concentration
        assign(ui_p.sub(idx), interpolate(ion['ki_init'], Wi.sub(idx).collapse()))
        assign(ue_p.sub(idx), interpolate(ion['ke_init'], We.sub(idx).collapse()))
        # add ion specific contribution to fraction alpha
        ui_prev_g = interpolate(ui_p.sub(idx), Wg)
        ue_prev_g = interpolate(ue_p.sub(idx), Wg)
        alpha_i_sum += Di*z*z*ui_prev_g
        alpha_e_sum += De*z*z*ue_prev_g
        
        # calculate and update Nernst potential for current ion
        ion['E'] = project(R*tmp/(F*z)*ln(ue_prev_g/ui_prev_g), Wg)
        # ion specific channel current
        ion['I_ch'] = phi_M_prev
        # add contribution to total channel current
        I_ch += ion['I_ch']

    J_phi_i = 0     # total intracellular flux
    J_phi_e = 0     # total extracellular flux

    # Initialize all parts of the variational form
    a00 = 0; a01 = 0; a02 = 0; L0 = 0
    a10 = 0; a11 = 0; a12 = 0; L1 = 0
    a20 = 0; a21 = 0; a22 = 0; L2 = 0

    # Setup ion specific part of variational formulation
    for idx, ion in enumerate(ion_list):
        # get ion attributes
        z = ion['z']; Di = ion['Di']; De = ion['De']; I_ch_k = ion['I_ch']

        # Set intracellular ion attributes
        ki = ui[idx]             # unknown
        ki_prev = ui_prev[idx]   # previous solution
        vki = vi[idx]            # test function
        # Set extracellular ion attributes
        ke = ue[idx]             # unknown
        ke_prev = ue_prev[idx]   # previous solution
        vke = ve[idx]            # test function
        # fraction of ion specific intra--and extracellular I_cap
        # Interpolate the previous solution on Gamma
        ki_prev_g = interpolate(ui_p.sub(idx), Wg)
        ke_prev_g = interpolate(ue_p.sub(idx), Wg)
        alpha_i = Di*z*z*ki_prev_g/alpha_i_sum
        alpha_e = De*z*z*ke_prev_g/alpha_e_sum

        # ion fluxes
        Ji = - Constant(Di)*grad(ki) - Constant(Di*z/psi)*ki_prev*grad(phi_i)  # linearised
        Je = - Constant(De)*grad(ke) - Constant(De*z/psi)*ke_prev*grad(phi_e)  # linearised
        
        # eq for k_i
        a00 += dt_inv*ki*vki*dxi - inner(Ji, grad(vki))*dxi
        a02 -= 1.0/(F*z)*alpha_i*p_IM*vki*dxGamma
        L0  += dt_inv*ki_prev*vki*dxi + 1.0/(F*z)*(I_ch_k - alpha_i*I_ch)*vki*dxGamma

        # eq for k_e
        a11 += dt_inv*ke*vke*dxe - inner(Je, grad(vke))*dxe
        a12 += 1.0/(F*z)*alpha_e*p_IM*vke*dxGamma
        L1  += dt_inv*ke_prev*vke*dxe - 1.0/(F*z)*(I_ch_k - alpha_e*I_ch)*vke*dxGamma

        # add contribution to total current flux
        J_phi_i += F*z*Ji
        J_phi_e += F*z*Je

        # MMS: add source terms
        L0 += inner(ion['f_k_i'], vki)*dxi # eq for k_i
        L1 += inner(ion['f_k_e'], vke)*dxe # eq for k_e
        # exterior boundary terms (zero in "physical" problem)
        L1 -= inner(dot(ion['J_k_e'], n_outer), vke)*dsOuter # eq for k_e
        L1 += F*z*inner(dot(ion['J_k_e'], n_outer), vphi_e)*dsOuter # eq for phi_e

    # equation for phi_i
    a00 += inner(J_phi_i, grad(vphi_i))*dxi
    a02 += inner(p_IM, vphi_i)*dxGamma

    # /int phi_i = 0: Lagrange terms
    a00 += _c*vphi_i*dxi + _d*phi_i*dxi

    # equation for phi_e
    a11 += inner(J_phi_e, grad(vphi_e))*dxe
    a12 -= inner(p_IM, vphi_e)*dxGamma

    # phi_M: Lagrange terms
    a20 += inner(phi_i, q_IM)*dxGamma
    a21 -= inner(phi_e, q_IM)*dxGamma
    a22 -= dt/C_M*inner(p_IM, q_IM)*dxGamma
    L2  += inner(phi_M_prev, q_IM)*dxGamma \
         - dt/C_M*inner(I_ch, q_IM)*dxGamma

    # add source term if MMS test
    L0 += inner(ion['phi_i_e'], _d)*dxi  # Lagrange for phi_i (if int phi_I != 0)
    L0 += inner(ion['f_phi_i'], vphi_i)*dxi    # eq for phi_i
    L1 += inner(ion['f_phi_e'], vphi_e)*dxe    # eq for phi_e

    # coupling condition IM = -Ji = Je + g
    L1 -= sum(inner(gM, vphi_e)*dxGamma(i) for i, gM in enumerate(ion['f_g_M'], 1))
    # eq for J_M
    L2 += dt/C_M*sum(inner(JM, q_IM)*dxGamma(i) for i, JM in enumerate(ion['f_J_M'], 1))
    
    # gather var form in matrix structure
    a = a00 + a01 + a02 + a10 + a11 + a12 + a20 + a21 + a22
    L = L0 + L1 + L2

    # -------------------------------- Solve ----------------------------------- #
    for k in range(int(round(Tstop/float(dt)))):

        wh = Function(W)
        solve(a == L, wh, solver_parameters={"linear_solver":"direct"})

        ui_p.assign(wh.sub(0))  # update ion specific membrane channels
        ue_p.assign(wh.sub(1))  # update ion specific membrane channels
        u_p.sub(2).assign(wh.sub(2))
        
        ## Export
        if do_export:
            encoding = XDMFFile.Encoding.HDF5 if has_hdf5() else XDMFFile.Encoding.ASCII
            if MPI.size(MPI.comm_world) > 1 and encoding == XDMFFile.Encoding.ASCII:
                print("XDMF file output not supported in parallel without HDF5")
            for i in range(N_ions+1):
                out_subi = XDMFFile(MPI.comm_world, "test-i"+str(i)+".xdmf")
                out_subi.write(ui_p.sub(i), encoding)
                out_sube = XDMFFile(MPI.comm_world, "test-e"+str(i)+".xdmf")
                out_sube.write(ue_p.sub(i), encoding)
            out_subg = XDMFFile(MPI.comm_world, "test-g.xdmf")
            out_subg.write(u_p.sub(2))

        # update previous membrane potential
        phi_M_prev.assign(interpolate(ui_p.sub(N_ions), Wg) \
                          - interpolate(ue_p.sub(N_ions), Wg))

        # updates problems time t
        t.assign(float(t + dt))

        # update Nernst potential for all ions
        for idx, ion in enumerate(ion_list):
            z = ion['z']
            ke_prev_g = interpolate(ue_p.sub(idx), Wg)
            ki_prev_g = interpolate(ui_p.sub(idx), Wg)
            ion['E'] = R*tmp/(F*z)*ln(ke_prev_g/ki_prev_g)

        """
        # DEBUG
        print("E_K", ion_list[1]['E'])
        print("E_Na", ion_list[0]['E'])
        print("total interior Na ", assemble(wh[0].sub(0)*dx))
        print("total exterior Na ", assemble(wh[1].sub(0)*dx))
        print("total Na ", assemble(wh[1].sub(0)*dx) + assemble(wh[0].sub(0)*dx))

        intrap = (0.35, 0.35)
        diff = project(wh[0].sub(0)(intrap) + wh[0].sub(1)(intrap) - wh[0].sub(2)(intrap), trace_space)
        print(diff.vector().array())
        """

    return wh, interior_mesh, exterior_mesh, gamma_mesh

n_values = [8, 16, 32, 64]
rNa_i = []
rNa_e = []
rK_i = []
rK_e = []
rCl_i = []
rCl_e = []
rphi_i = []
rphi_e = []
rJM = []
# For L2 errors
ENa_i_L2 = []
ENa_e_L2 = []
EK_i_L2 = []
EK_e_L2 = []
ECl_i_L2 = []
ECl_e_L2 = []
Ephi_i_L2 = []
Ephi_e_L2 = []
EJM_L2 = []
# For H1 errors
ENa_i_H1 = []
ENa_e_H1 = []
EK_i_H1 = []
EK_e_H1 = []
ECl_i_H1 = []
ECl_e_H1 = []
Ephi_i_H1 = []
Ephi_e_H1 = []

for i in range(len(n_values)):

    n = n_values[i]
    # time variables
    t = Constant(0.0)
    dt = 1.0e-5/(n*n) # time step
    Tstop = 1.0e-7    # end time

    # get MMS terms and exact solutions
    M = MMS()
    src_terms, exact_sols, init_conds, bndry_terms, \
            subdomains_MMS =  M.get_MMS_terms_KNPEMI(t)

    Na = {'Di':1.0, 'De':1.0, 'z':1.0,
          'ki_init':init_conds['Na_i'],
          'ke_init':init_conds['Na_e'],
          'f_k_i':src_terms['f_Na_i'],
          'f_k_e':src_terms['f_Na_e'],
          'J_k_e':bndry_terms['J_Na_e'],
          'phi_i_e':exact_sols['phi_i_e'],
          'f_phi_i':src_terms['f_phi_i'],
          'f_phi_e':src_terms['f_phi_e'],
          'f_g_M':src_terms['f_g_M'],
          'f_J_M':src_terms['f_J_M']}

    K = {'Di':1.0, 'De':1.0, 'z':1.0,
         'ki_init':init_conds['K_i'],
         'ke_init':init_conds['K_e'],
         'f_k_i':src_terms['f_K_i'],
         'f_k_e':src_terms['f_K_e'],
         'J_k_e':bndry_terms['J_K_e'],
         'phi_i_e':exact_sols['phi_i_e'],
         'f_phi_i':src_terms['f_phi_i'],
         'f_phi_e':src_terms['f_phi_e'],
         'f_g_M':src_terms['f_g_M'],
         'f_J_M':src_terms['f_J_M']}

    Cl = {'Di':1.0, 'De':1.0, 'z':-1.0,
          'ki_init':init_conds['Cl_i'],
          'ke_init':init_conds['Cl_e'],
          'f_k_i':src_terms['f_Cl_i'],
          'f_k_e':src_terms['f_Cl_e'],
          'J_k_e':bndry_terms['J_Cl_e'],
          'phi_i_e':exact_sols['phi_i_e'],
          'f_phi_i':src_terms['f_phi_i'],
          'f_phi_e':src_terms['f_phi_e'],
          'f_g_M':src_terms['f_g_M'],
          'f_J_M':src_terms['f_J_M']}

    ion_list = [Na, K, Cl]
    # solve
    wh, interior_mesh, exterior_mesh, gamma_mesh = solve_system(n, t, dt, Tstop, ion_list, M, order)

    hi = interior_mesh.hmin()
    he = exterior_mesh.hmin()
    hg = gamma_mesh.hmin()
    # Intracellular
    Na_i = wh.sub(0).sub(0)     # Na concentration
    K_i = wh.sub(0).sub(1)      # K concentration
    Cl_i = wh.sub(0).sub(2)     # Cl concentration
    phi_i = wh.sub(0).sub(3)    # potential
    # Extracellular
    Na_e = wh.sub(1).sub(0)     # Na concentration
    K_e = wh.sub(1).sub(1)      # K concentration
    Cl_e = wh.sub(1).sub(2)     # Cl concentration
    phi_e = wh.sub(1).sub(3)    # potential
    # Membrane
    J_M = wh.sub(2)             # potential

    # print("-----------------------------------------------")
    # print("-----------------------------------------------")
    # print("JM")
    # print(J_M.vector().str(True))
    # print("-----------------------------------------------")
    # print("-----------------------------------------------")

    # function space for exact solutions
    VI = FiniteElement('CG', interior_mesh.ufl_cell(), 4)   # define element
    VI = FunctionSpace(interior_mesh, VI)                   # define function space
    VE = FiniteElement('CG', exterior_mesh.ufl_cell(), 4)   # define element
    VE = FunctionSpace(exterior_mesh, VE)                   # define function space
    VG = FiniteElement('CG', gamma_mesh.ufl_cell(), 4)      # define element
    VG = FunctionSpace(gamma_mesh, VG)                      # define function space

    Na_i_e = interpolate(exact_sols['Na_i_e'], VI)         # Na intracellular
    Na_e_e = interpolate(exact_sols['Na_e_e'], VE)         # Na extracellular
    K_i_e = interpolate(exact_sols['K_i_e'], VI)           # K intracellular
    K_e_e = interpolate(exact_sols['K_e_e'], VE)           # K extracellular
    Cl_i_e = interpolate(exact_sols['Cl_i_e'], VI)         # Cl intracellular
    Cl_e_e = interpolate(exact_sols['Cl_e_e'], VE)         # Cl extracellular
    phi_i_e = interpolate(exact_sols['phi_i_e'], VI)       # phi intracellular
    phi_e_e = interpolate(exact_sols['phi_e_e'], VE)       # phi extracellular

    ## Export exact solution
    if do_export:
        encoding = XDMFFile.Encoding.HDF5 if has_hdf5() else XDMFFile.Encoding.ASCII
        if MPI.size(MPI.comm_world) > 1 and encoding == XDMFFile.Encoding.ASCII:
            print("XDMF file output not supported in parallel without HDF5")        
        out_nai = XDMFFile(MPI.comm_world, "exact-Na_i.xdmf")
        out_nai.write(Na_i_e)
        out_nae = XDMFFile(MPI.comm_world, "exact-Na_e.xdmf")
        out_nae.write(Na_e_e)
        out_kii = XDMFFile(MPI.comm_world, "exact-Ki_i.xdmf")
        out_kii.write(K_i_e)
        out_kie = XDMFFile(MPI.comm_world, "exact-Ki_e.xdmf")
        out_kie.write(K_e_e)
        out_cli = XDMFFile(MPI.comm_world, "exact-Cl_i.xdmf")
        out_cli.write(Cl_i_e)
        out_cle = XDMFFile(MPI.comm_world, "exact-Cl_e.xdmf")
        out_cle.write(Cl_e_e)
        out_phii = XDMFFile(MPI.comm_world, "exact-phi_i.xdmf")
        out_phii.write(phi_i_e)
        out_phie = XDMFFile(MPI.comm_world, "exact-phi_e.xdmf")
        out_phie.write(phi_e_e)

    # get error L2
    Nai_L2 = errornorm(Na_i_e, Na_i, "L2", degree_rise=4)
    Nae_L2 = errornorm(Na_e_e, Na_e, "L2", degree_rise=4)
    Ki_L2 = errornorm(K_i_e, K_i, "L2", degree_rise=4)
    Ke_L2 = errornorm(K_e_e, K_e, "L2", degree_rise=4)
    Cli_L2 = errornorm(Cl_i_e, Cl_i, "L2", degree_rise=4)
    Cle_L2 = errornorm(Cl_e_e, Cl_e, "L2", degree_rise=4)
    phii_L2 = errornorm(phi_i_e, phi_i, "L2", degree_rise=4)
    phie_L2 = errornorm(phi_e_e, phi_e, "L2", degree_rise=4)

    ENa_i_L2.append(Nai_L2)
    ENa_e_L2.append(Nae_L2)
    EK_i_L2.append(Ki_L2)
    EK_e_L2.append(Ke_L2)
    ECl_i_L2.append(Cli_L2)
    ECl_e_L2.append(Cle_L2)
    Ephi_i_L2.append(phii_L2)
    Ephi_e_L2.append(phie_L2)

    # get error H1
    Nai_H1 = errornorm(Na_i_e, Na_i, "H1", degree_rise=4)
    Nae_H1 = errornorm(Na_e_e, Na_e, "H1", degree_rise=4)
    Ki_H1 = errornorm(K_i_e, K_i, "H1", degree_rise=4)
    Ke_H1 = errornorm(K_e_e, K_e, "H1", degree_rise=4)
    Cli_H1 = errornorm(Cl_i_e, Cl_i, "H1", degree_rise=4)
    Cle_H1 = errornorm(Cl_e_e, Cl_e, "H1", degree_rise=4)
    phii_H1 = errornorm(phi_i_e, phi_i, "H1", degree_rise=4)
    phie_H1 = errornorm(phi_e_e, phi_e, "H1", degree_rise=4)

    ENa_i_H1.append(Nai_H1)
    ENa_e_H1.append(Nae_H1)
    EK_i_H1.append(Ki_H1)
    EK_e_H1.append(Ke_H1)
    ECl_i_H1.append(Cli_H1)
    ECl_e_H1.append(Cle_H1)
    Ephi_i_H1.append(phii_H1)
    Ephi_e_H1.append(phie_H1)

    if i > 0:
        # L2 errors
        r_NaI_L2 = np.log(Nai_L2/Nai_L2_0)/np.log(hi/hi0)
        r_NaE_L2 = np.log(Nae_L2/Nae_L2_0)/np.log(he/he0)
        r_KI_L2 = np.log(Ki_L2/Ki_L2_0)/np.log(hi/hi0)
        r_KE_L2 = np.log(Ke_L2/Ke_L2_0)/np.log(he/he0)
        r_ClI_L2 = np.log(Cli_L2/Cli_L2_0)/np.log(hi/hi0)
        r_ClE_L2 = np.log(Cle_L2/Cle_L2_0)/np.log(he/he0)
        r_phiI_L2 = np.log(phii_L2/phii_L2_0)/np.log(hi/hi0)
        r_phiE_L2 = np.log(phie_L2/phie_L2_0)/np.log(he/he0)

        rNa_i.append(r_NaI_L2)
        rNa_e.append(r_NaE_L2)
        rK_i.append(r_KI_L2)
        rK_e.append(r_KE_L2)
        rCl_i.append(r_ClI_L2)
        rCl_e.append(r_ClE_L2)
        rphi_i.append(r_phiI_L2)
        rphi_e.append(r_phiE_L2)

    # update prev h
    hi0 = hi
    he0 = he
    hg0 = hg
    # update prev L2
    Nai_L2_0, Nae_L2_0 = Nai_L2, Nae_L2
    Ki_L2_0, Ke_L2_0 = Ki_L2, Ke_L2
    Cli_L2_0, Cle_L2_0 = Cli_L2, Cle_L2
    phii_L2_0, phie_L2_0 = phii_L2, phie_L2
    ## JM_L2_0 = JM_L2

print("Rate(L2) Na I: ", rNa_i)
print("Rate(L2) Na E: ",rNa_e)
print("Rate(L2) K I: ",rK_i)
print("Rate(L2) K E: ",rK_e)
print("Rate(L2) Cl I: ",rCl_i)
print("Rate(L2) Cl E: ",rCl_e)
print("Rate(L2) phi I: ",rphi_i)
print("Rate(L2) phi E: ",rphi_e)
print("Rate(L2) JM: ",rJM)

## Print error data to files
nai_data = open("Na_i_P"+str(order)+".dat","w+")
nai_data.write( "N\tL2\tH1\n" )
nae_data = open("Na_e_P"+str(order)+".dat","w+")
nae_data.write( "N\tL2\tH1\n" )
ki_data = open("K_i_P"+str(order)+".dat","w+")
ki_data.write( "N\tL2\tH1\n" )
ke_data = open("K_e_P"+str(order)+".dat","w+")
ke_data.write( "N\tL2\tH1\n" )
cli_data = open("Cl_i_P"+str(order)+".dat","w+")
cli_data.write( "N\tL2\tH1\n" )
cle_data = open("Cl_e_P"+str(order)+".dat","w+")
cle_data.write( "N\tL2\tH1\n" )
phii_data = open("Phi_i_P"+str(order)+".dat","w+")
phii_data.write( "N\tL2\tH1\n" )
phie_data = open("Phi_e_P"+str(order)+".dat","w+")
phie_data.write( "N\tL2\tH1\n" )

for i in range(len(n_values)):
    nai_data.write( str(n_values[i]) + "\t" + str(ENa_i_L2[i]) + "\t" + str(ENa_i_H1[i]) + "\n" )
    nae_data.write( str(n_values[i]) + "\t" + str(ENa_e_L2[i]) + "\t" + str(ENa_e_H1[i]) + "\n" )
    ki_data.write( str(n_values[i]) + "\t" + str(EK_i_L2[i]) + "\t" + str(EK_i_H1[i]) + "\n" )
    ke_data.write( str(n_values[i]) + "\t" + str(EK_e_L2[i]) + "\t" + str(EK_e_H1[i]) + "\n" )
    cli_data.write( str(n_values[i]) + "\t" + str(ECl_i_L2[i]) + "\t" + str(ECl_i_H1[i]) + "\n" )
    cle_data.write( str(n_values[i]) + "\t" + str(ECl_e_L2[i]) + "\t" + str(ECl_e_H1[i]) + "\n" )
    phii_data.write( str(n_values[i]) + "\t" + str(Ephi_i_L2[i]) + "\t" + str(Ephi_i_H1[i]) + "\n" )
    phie_data.write( str(n_values[i]) + "\t" + str(Ephi_e_L2[i]) + "\t" + str(Ephi_e_H1[i]) + "\n" )


