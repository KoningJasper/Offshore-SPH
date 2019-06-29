# Add parent folder to path
import sys, os, math
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import numba
import h5py
import scipy.linalg

from src.Solver import Solver
from src.Helpers import Helpers
from src.Methods.WCSPH import WCSPH
from src.Kernels.Wendland import Wendland
from src.Integrators.Verlet import Verlet
from src.Integrators.PEC import PEC
from src.Integrators.NewmarkBeta import NewmarkBeta
from src.Post.Plot import Plot
from src.Common import ParticleType, particle_dtype
from src.Equations.SummationDensity import SummationDensity
from src.Equations.TaitEOS import TaitEOS
from src.Equations.Shepard import Shepard

def create_particles(Nx: int, rho0: float, x_end: float, y_end: float, ice_max: float, P: dict, compression: float = 1.0):
    """
        Nx: int
            Number of particles in x-direction. Total number of particles is computed by assuming a constant spacing (r0).
    """
    r0   = x_end / Nx # Distance between particles.

    # Create some fluid particles
    fluid = Helpers.rect(xmin=0, xmax=x_end, ymin=0, ymax=y_end, r0=r0, mass=1.0, rho0=rho0, pack=False)

    # Compute the mass based on the average area
    dA         = x_end * y_end / len(fluid)
    mass       = dA * rho0
    fluid['m'] = mass

    # Compress the fluid vertically
    fluid['y'] = fluid['y'] / compression
    y_end      = y_end / compression

    # Maximum and minimum values of boundaries
    x_min = - r0
    x_max = x_end + r0
    y_min = - r0
    y_max = 1.3 * y_end + r0
    
    # Create the boundary
    bottom = Helpers.rect(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_min, r0=r0, mass=0., rho0=0., label=ParticleType.Boundary)
    left   = Helpers.rect(xmin=x_min, xmax=x_min, ymin=y_min, ymax=y_max, r0=r0, mass=0., rho0=0., label=ParticleType.Boundary)
    right  = Helpers.rect(xmin=x_max, xmax=x_max, ymin=y_min, ymax=y_max, r0=r0, mass=0., rho0=0., label=ParticleType.Boundary)

    # Create ice-layer
    yy = fluid['y'].max() + r0
    if P['model'] == 0:
        # Above the water
        yy = yy + 2.0
        ice = ice_layer(r0, x_min, ice_max, yy, mass, rho0)
    elif P['model'] == 1:
        # Float in the middle.
        ice = ice_layer(r0, x_min, x_max, yy, mass, rho0)
    elif P['model'] in [2, 4, 5, 6]:
        # Regular ice-sheet.
        # With extra 
        if P['model'] == 4:
            yy = yy + 2.0 # Make it above the water.
        lyr = ice_layer(r0, x_min, ice_max * 1.9, yy, mass, rho0)
        xtra = Helpers.rect(xmin=ice_max * 1.9 + r0, xmax=x_max, ymin=yy, ymax=yy, r0=r0, mass=0., rho0=0., label=ParticleType.Boundary)
        ice = np.concatenate((lyr, xtra))
    else:
        raise 'Invalid model.'
    
    # Return the particle array, and extra properties.
    return yy, r0, np.concatenate((fluid, bottom, left, right, ice))

def ice_layer(r0: float, xmin: float, xmax: float, y: float, mass: float, rho0: float):
    """ Creates a coupled particle-layer; representing the ice-sheet. """
    return Helpers.rect(xmin=xmin, xmax=xmax, ymin=y, ymax=y, r0=r0, mass=mass, rho0=rho0, label=ParticleType.Coupled)

def createMatrix(n: int, E: float, I: float, A: float, rho: float, L_n: float):
    """
        Create matrices for a clamped-free beam.

        Returns:
            K: np.array
                N x N, matrix containing stiffness
            M: np.array
                N x N, identiy matrix containing mass
    """
    # Initialize full FD-matrix
    stamp = np.array([1, -4, 6, -4, 1]) # Central difference stamp.
    fd_coef = np.zeros((n, n + 4))
    for i in range(n):
        fd_coef[i, i:i + len(stamp)] = stamp

    # Apply BC
    fd_coef = fd_coef[:, 2:-2]
    fd_coef[0, 0] = 7; fd_coef[n - 1, n - 1] = 2; fd_coef[n - 1, n - 3] = 2
    fd_coef[n - 2, n - 1] = -2; fd_coef[n - 2, n - 2] = 5

    # Construct K matrix
    K = (E * I / L_n ** 4) * fd_coef

    # Construct the mass matrix.
    M = np.eye(n,n) * rho * A

    return K, M
    
@numba.jit()
def coupling(pA: np.array, solver: Solver) -> np.array:
    P = solver.couplingProperties
    t = solver.t - solver.settleTime # Actual simulation time.
    
    # Extract state data
    cInd = (pA['label'] == ParticleType.Coupled)
    y    = pA[cInd]['y'] # GCS
    w    = y - P['y0']   # LCS
    wdot = pA[cInd]['vy']
    x    = pA[cInd]['x']

    # Find the left most fluid particle.
    xmin = pA['x'][solver.f_indexes].min() 

    # Determine the pressure; using SPH interpolation
    P_n = np.zeros_like(w)
    rho_n = np.copy(pA['rho'][cInd])
    if P['model'] > 0:
        for i in range(len(w)):
            if x[i] < xmin:
                continue
            else:
                rho_n[i], P_n[i] = pressure_SPH(x[i], y[i], pA, solver, P)
    F_n = P_n * P['b'] # Force (per m) acting on center of block, upward.

    if solver.customSettle == None:
        # Only store when not using custom settle, otherwise custom settle already stores the information.
        P['Pressure'].append(P_n)

    # Determine (vertical) force on ice
    Fw = np.zeros_like(w)
    if P['model'] in [5, 6] and solver.settled == False:
        # Ice-structure force, don't force when settling.
        _, Fw_last = iceForce(t, w, P)
        if w[-1] > 0:
            Fw_last[1] = - w[-1] * P['m2_stiffness'] * P['b']

        # Determine the mode
        iceMode(t, solver.dt, P)

        # Only Y-force, only on the most right node, and convert to N/m
        Fw[-1] = Fw_last[1] / P['L_n'] 
    elif P['model'] in [0]:
        # Static force for model 0 (N/m)
        Fw[-1] = 1e4 / P['L_n']
    
    if P['model'] in [1, 2, 4, 5, 6]:
        # Add gravity in case of model >1, downward. (N per meter)
        Fw = Fw - P['m'] * P['g']

        # Add upward force on the left, to prevent rotation of the sheet, equal to w * rho_w * g
        F_r = np.zeros_like(w)
        for i in range(len(w)):
            if x[i] >= xmin:
                continue
            else:
                F_r[i] = - w[i] * solver.method.rho0 * P['g']
        P['F_r'].append(F_r)
        Fw = Fw + F_r
    
    # Compute accelerations
    a = solver.couplingIntegrator.acceleration(solver.dt, Fw + F_n, w, wdot)
    
    # Set the particle array
    pA['ay'][cInd]  = a
    pA['p'][cInd]   = P_n
    pA['rho'][cInd] = rho_n
    
    return pA

@numba.jit()
def iceForce(t: float, w: np.array, P: dict):
    """
        Computes the force in newtons exerted on the ice by the hull.
        
        Parameters
        ----------
        t: float
            current time [s].
        w: np.array
            y-coordinate of the ice-sheet [m].
        P: dict
            Properties
            
        Returns
        -------
        P: dict
            Properties
        F: np.array
            Force on the node at the hull due to the hull forces.
    """
    # Compute interface length and penetration
    ice_x       = P['ice_v'] * t
    perp        = np.sin(P['alpha']) * ice_x + np.cos(P['alpha']) * w[-1]
    L_interface = perp * (np.tan(P['alpha']) + 1 / np.tan(P['alpha']))
    
    # Store the penetration & interface length for later retrieval
    P['penetration'].append(perp)
    P['L_interface'].append(L_interface)
    
    # Compute normal
    vsl_hull_normal = - P['alpha'] - np.pi / 2
    vsl_hull_n = np.array([np.cos(vsl_hull_normal), np.sin(vsl_hull_normal)])
    
    m = P['mode']
    if m == 1:
        # Mode 1; crushing
        fc_amp = P['b'] * P['ice_fy_comp'] * L_interface
    elif m == 2:
        # Mode 2; sliding
        if L_interface < P['m2_ref_l']:
            fc_amp = (P['m2_ref_force'] / vsl_hull_n[1]) - P['b'] * P['m2_stiffness'] * (P['m2_ref_l'] - L_interface)
        else:
            # Avoid large forces around the transition.
            fc_amp = P['b'] * P['ice_fy_comp'] * L_interface
    
    # Compute force, with direction.
    F = fc_amp * vsl_hull_n
    P['F_a'].append(F[1]) # Store the vertical force
    return P, F

@numba.jit()
def iceMode(t: float, dt: float, P: dict):
    # Transition free period
    if (t - P['t_trans']) < 0.01:
        return P
    
    v_pen = np.diff(P['penetration']) / dt # Speed of penetration [m/s]
    v_pen = v_pen[-1] # Only the last speed.
    
    if P['mode'] == 1 and v_pen < 0:
        P['m2_ref_l'] = P['L_interface'][-1]
        P['m2_ref_force'] = P['F_a'][-1]
        P['t_trans'] = t
        P['mode'] = 2
        return P
    elif P['mode'] == 2 and P['F_a'][-1] < P['m2_ref_force'] and v_pen >= 0:
        P['t_trans'] = t
        P['mode'] = 1
        return P
    return P

@numba.jit()
def pressure_SPH(x: float, y: float, pA: np.array, solver: Solver, P: dict):
    """ 
        Uses SPH to interpolate the pressure at the given location (x, y).
        Summation density is used to estimate the density at the location, 
        after which Tait EOS is used to determine the pressure at the given location.

        Parameters
        ----------
        x: float
            x-coordinate of the particle to find the pressure at.
        y: float
            y-coordinate of the particle to find the pressure at.
        pA: np.array
            Complete particle array.
    """
    h = P['h']

    # Find the near particles
    h_i, _, dist, near_arr = solver.nn.nearPos(x, y - P['hh'], h, pA)

    # Compute kernel value
    w = solver.kernel.evaluate(dist, np.ones_like(h_i) * h)

    # Correct using shepard filter.
    w_tilde = Shepard(w, pA[near_arr]['label'], pA[near_arr]['m'], pA[near_arr]['rho'])

    # Compute summation density at the location
    rho = SummationDensity(pA[near_arr]['label'], pA[near_arr]['m'], w_tilde)

    # Compute the pressure using TaitEOS
    ratio = (rho / solver.method.rho0) ** solver.method.gamma
    p = (ratio - 1.0) * solver.method.B
    return rho, p

@numba.jit()
def custom_settling(pA: np.array, solver: Solver) -> bool:
    """
        Determines if the coupling has settled.

        Parameters
        ----------
        pA: np.array
            Particle array

        Returns
        -------
        settled: bool
            Has the coupling settled?
    """
    # Just wait the 1800 time-steps.
    return False
    # P = solver.couplingProperties

    # # Extract state data
    # cInd = (pA['label'] == ParticleType.Coupled)
    # y    = pA[cInd]['y'] # GCS
    # w    = y - P['y0']   # LCS
    # x    = pA[cInd]['x']

    # # Find the left most fluid particle.
    # xmin = pA['x'][solver.f_indexes].min() 

    # # Determine the pressure; using SPH interpolation
    # P_n = np.zeros_like(w)
    # rho_n = np.copy(pA['rho'][cInd])
    # for i in range(len(w)):
    #     if x[i] < xmin:
    #         continue
    #     else:
    #         rho_n[i], P_n[i] = pressure_SPH(x[i], y[i], pA, solver, P)
    # P['Pressure'].append(P_n)

    # # Compute the mean pressure over-time, at at least a 200 time-steps.
    # if len(P['Pressure']) > 200:
    #     meanPressure = np.array(P['Pressure'][-200:]).mean(axis=1) / 1e3

    #     # See if the max and min differ a maximum of 2 kPa from mean of mean
    #     mmean = meanPressure.mean()
    #     if (abs(mmean - meanPressure.max()) <= 2) and (abs(mmean - meanPressure.min()) <= 2):
    #         return True
    #     else:
    #         return False
    # else:
    #     return False

@numba.jit()
def angle(w: np.array):
    """
        Computes the angle of the beam.

        Parameters
        ----------
        w: np.array
            y-coordinate of the ice-sheet

        Returns
        -------
        phi: np.array
            Array of angles for each elements.
    """
    # Initialize full matrix
    stamp = [-1/2, 0, 1/2]
    mat = np.zeros((len(w), len(w) + 2))
    for i in range(len(w)):
        mat[i, i:i + len(stamp)] = stamp

    # BCs
    mat = mat[:, 1:-1]
    mat[0, 0] = -1; mat[0, 1] = 1
    mat[len(w) - 1, len(w) - 1] = 1; mat[len(w) - 1, len(w) - 2] = -1
    
    # Compute the angle
    phi = np.dot(mat, w)
    return phi

def iceStress(w: np.array, F: np.array, E: float, h: float, v: float, L_n: float):
    """
        Computes the stress in the beam/ice-sheet for a given w
        
        Parameters
        ----------
        w: np.array
            y-coordinate positions of the ice-sheet
        F: np.array
            x and y force on the ice-sheet.
            
        Returns
        -------
        stress: np.array
            the stress at each nodal location of the ice-sheet.
    """
    
    # Initialize the FD-matrix
    stamp = [1, -2, 1]
    mat = np.zeros((len(w) - 2, len(w)))
    for i in range(len(w) - 2):
        mat[i, i:i + len(stamp)] = stamp
        
    # Compute w'' ; w''(0) = 0 and w''(L) = 0
    dw2 = np.zeros(len(w))
    dw2[1:-1] = np.dot(mat, w) / (L_n ** 2)
    
    stress = np.abs(E * h / (2 * (1 - v ** 2)) * dw2) - F[0] / h
    return stress
    
def natFreq(M: np.array, K: np.array, n: int):
    """
        Compute the n-th natural frequency of the system.
        
        Parameters
        ----------
        M: np.array
            Mass matrix
        K: np.array
            Stiffness matrix
        n: int
            N-th natural frequency to return.
            
        Returns
        -------
        omega: float
            The n-th natural frequency of the system.
    """
    
    # Compute eigenvector and eigenvalue.
    w, v = scipy.linalg.eig(M, K)
    
    # Compute frequency
    omega = 1 / np.sqrt(np.real(w))
    omega = omega / (2 * np.pi)
    
    return omega[n]
    
def post(file: str):
    with h5py.File(file, 'r') as h5f:
        pA = h5f['particleArray'][:]
        x = h5f['x'][:]
        y = h5f['y'][:]
        dt = h5f['dt_a'][:]
        
        # Read some scalars
        L_n = h5f['L_n'][()]; m = h5f['m'][()]; b = h5f['b'][()]
        yy = h5f['y0'][()]; v_ice = h5f['ice_v'][()]; v = h5f['pois'][()]
        h = h5f['hh'][()] * 2; E = h5f['E'][()]; stl = h5f['settleTime'][()]

    s = []; cIndex = pA['label'] == 3
    fl = 5e5; xmin = 45; xmax = x[0, cIndex].max(); tt = 0.0
    for i in range(len(y)):
        w = y[i, cIndex] - yy
        xx = x[i, cIndex]
        
        # Filter w for only > xmin
        w_f = []; x_f = []
        for n in range(len(w)):
            if xx[n] >= xmin and xx[n] <= (xmax - 7):
                w_f.append(w[n]); x_f.append(xx[n])
        w_f = np.array(w_f)
        
        s_a = iceStress(w_f, np.zeros_like(w_f), E, h, v, L_n)
        
        if s_a.max() > fl and tt > (0.1 + stl):
            t = dt[0:i].sum() - stl
            LL = np.array(x_f).max() - x_f[s_a.argmax()] + 7
            print('V_ice: {0}'.format(v_ice))
            print('Failed @ t = {0:f} s'.format(t))
            print('Breaking Length: {0:f} m'.format(LL))
            break
        s.append(s_a)
        tt += dt[i]

    with open('BL.txt', 'a') as f:
        f.write('({0}, {1}, {2}, {3})\n'.format(v_ice, t, LL, cIndex.sum()))

def main():
    # ----- Setup ----- #
    # Simulation parameters
    duration = 5.0  # Duration of the simulation [s]
    height   = 10.0  # Height of the fluid box [m]
    width    = 60.0  # Width of the fluid box [m]
    compress = 1.0   # (vertical) compression factor [-]

    # Computed parameters
    ice_max = width / 2.0

    # Coupling object
    P = { 'g': 9.81, 'L_interface': [], 'F_a': [], 'Pressure': [], 'F_r': []}
    
    # Other parameters
    rho0 = 1025.0
    XSPH = True
    print('Three different models are implemented.')
    print('0. A cantilever beam, with a static force on the end, above the water.')
    print('1. An short sheet of ice (cantilever beam) floating on water without any force.')
    print('2. An long sheet of ice (cantilever beam) floating on water without any force.')
    print('3. --')
    print('4. The ice-sheet above the water.')
    print('5. The scaled ice-sheet model (Valanto, 1992).')
    print('6. Full scale model (Keijdener, 2018).')
    P['model'] = int(input('Select model: '))
    plot = input('Create animation after solving? (y/n) ').lower() == 'y'

    if P['model'] > 6 or P['model'] < 0:
        raise 'Invalid model selected'
    
    if P['model'] == 5:
        height = 1.0; width = 2.0
        ice_max = 1.0

    Nx = int(input('Number of particles in x-direction (100 > N > 10): '))
    # if Nx > 500 or Nx < 10:
        # raise 'Invalid number of particles.'
    
    # Create some particles
    y00, r0, pA = create_particles(Nx, rho0, width, height, ice_max, P, compress)
    P['y0'] = y00

    # -- Create matrix -- #
    # Beam/Ice properties
    L = ice_max + r0              # Total Length [m]; one third overlap with water, two thirds to the left.
    b = 1                         # Width of the beam [m]
    h = 1                         # Height of the beam [m]
    v = 0.3                       # Poisson ratio [-]
    P['alpha'] = 15 / 180 * np.pi # Angle of the hull [rad]
    P['ice_v'] = 0.2              # Velocity of the ice-sheet [m/s]
    P['ice_fy_comp'] = 11e3       # Compressive strength of the ice [Pa]
    P['m2_stiffness'] = 15e5      # Rigid spring stiffness [N/m]
    P['b'] = b

    # General properties
    if P['model'] in [0]:
        E = 200e9; rho = 7800
        h = 1/33.33
    elif P['model'] in [1, 2]:
        E = 140e6; rho = 916.0; duration = 1.0
        if P['model'] == 1:
            L = 2 * ice_max 
        elif P['model'] == 2:
            L = 5 * ice_max
    elif P['model'] in [5]:
        duration = 1.5
        L   = 5 * ice_max
        h   = 1/33.33
        b   = 0.34; P['b'] = b
        E   = 140e6 # Young's Modulus [Pa]
        rho = 916.0
    elif P['model'] in [6]:
        L = 1.9 * ice_max
        P['ice_v'] = float(input('Ice velocity: '))
        duration = float(input('Duration: '))
        E = 5e9 / (1 - v ** 2); P['ice_fy_comp'] = 6E5
        P['m2_stiffness'] = 50 * P['ice_fy_comp']
        P['alpha'] = 45 / 180 * np.pi
        rho = 925.0
    
    # Ice-penetration
    P['mode'] = 1          # Start in crushing mode.
    P['penetration'] = [0] # Start with zero penetration.
    P['t_trans'] = 0       # Transition time [s]
    P['hh'] = h / 2

    # Compute one-time dynamic h.
    P['h'] = 1.6 * r0

    # Compute properties
    n = len(pA[pA['label'] == ParticleType.Coupled]) # Number of Nodes [-]
    A = b * h                                        # Area [m^3]
    I = 1 / 12 * b * h ** 3                          # Area moment of Inertia [kg/m^3]
    P['L_n'] = L / n                                 # Node length [m]
    P['m'] = rho * A                                 # Nodal mass [kg/m]

    # Assign some props
    P['E'] = E
    P['pois'] = v

    # Compute stiffness and mass matrices
    K, M = createMatrix(n, E, I, A, rho, P['L_n'])

    # Set mass
    pA[pA['label'] == ParticleType.Coupled]['m'] = P['m']

    # Compute damping matrix
    xi     = 2e1                              # Damping factor [-]
    c_crit = np.sqrt(rho * A * rho0 * P['g']) # Critical damping factor [-]
    C      = np.eye(n) * 2 * xi * c_crit      # Damping matrix [N/s]

    # Create the solver
    newmark    = NewmarkBeta(beta=1/4, gamma=1/2, M=M, K=K, C=C)
    kernel     = Wendland()
    method     = WCSPH(height=height, rho0=rho0, r0=r0, useXSPH=XSPH, Pb=0, useSummationDensity=False)
    integrator = PEC(useXSPH=XSPH, strict=False)
    solver     = Solver(
                    method, integrator, kernel, duration, quick=False, damping=0.0,
                    incrementalWriteout=False, maxSettle=1800, customSettle=custom_settling, exportProperties=['x', 'y', 'p', 'rho'], 
                    coupling=coupling, couplingIntegrator=newmark, couplingProperties=P, timeStep=None
                )

    # Add the particles
    solver.addParticles(pA)

    # Setup
    solver.setup()

    # ----- Run ----- #
    solver.run()

    # ----- Post ----- #
    # Output timing
    solver.timing()

    # Output
    exportPath = '{0}/IceBreak-{1}.hdf5'.format(sys.path[0], P['model'])
    solver.save(exportPath, True, solver.couplingProperties)

    # Create an animation.
    if plot == True:
        title = 'IceBreak (2D); {0} particles'.format(len(pA))
        plt = Plot(exportPath, title=title, xmin=-1, xmax=width + 2 * r0, ymin=- 2 * r0, ymax=height * 1.3 + 2 * r0)
        plt.save('{0}/IceBreak-{1}.mp4'.format(sys.path[0], P['model']))

    # Load the file and determine breaking length
    post(exportPath)

if __name__ == '__main__':
    main()
