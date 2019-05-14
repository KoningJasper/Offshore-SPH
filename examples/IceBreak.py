# Add parent folder to path
import sys, os, math
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import numba

from src.Solver import Solver
from src.Helpers import Helpers
from src.Methods.WCSPH import WCSPH
from src.Kernels.Wendland import Wendland
from src.Integrators.Verlet import Verlet
from src.Integrators.PEC import PEC
from src.Post.Plot import Plot
from src.Common import ParticleType, particle_dtype
from src.Equations.SummationDensity import SummationDensity
from src.Equations.TaitEOS import TaitEOS

def create_particles(Nx: int, rho0: float, x_end: float, y_end: float, ice_max: float):
    """
        Nx: int
            Number of particles in x-direction. Total number of particles is computed by assuming a constant spacing (r0).
    """
    r0   = x_end / Nx # Distance between particles.

    # Create some fluid particles
    fluid = Helpers.rect(xmin=0, xmax=x_end, ymin=0, ymax=y_end, r0=r0, mass=1.0, rho0=rho0, pack=True)

    # Compute the mass based on the average area
    dA         = x_end * y_end / len(fluid)
    mass       = dA * rho0
    fluid['m'] = mass

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
    global yy
    yy = fluid['y'].max() + r0
    ice = ice_layer(r0, - ice_max * 2, ice_max, yy)
    return yy, r0, np.concatenate((fluid, bottom, left, right, ice))
    # return r0, np.concatenate((fluid, bottom, left, right))

def ice_layer(r0: float, xmin: float, xmax: float, y: float):
    """ Creates a coupled particle-layer; representing the ice-sheet. """
    return Helpers.rect(xmin=xmin, xmax=xmax, ymin=y, ymax=y, r0=r0, mass=0., rho0=0., label=ParticleType.Coupled)

def createMatrix(n: int, E: float, I: float, A: float, rho: float, L: float):
    """
        Create matrices for a free-free beam.

        Returns:
            K: np.array
                N x N, matrix containing stiffness
            M: np.array
                N x N, identiy matrix containing mass
    """
    # Central difference stamp.
    stamp = np.array([1, -4, 6, -4, 1])
    order = 2

    # Initialize full matrix
    fmat = np.zeros((n, n + 2 * order))
    for i in range(n):
        fmat[i, i:i + len(stamp)] = stamp

    # Apply BC
    nmat = fmat[:, 2:-2]
    n_i = n - 1 # Index based

    # Set left & right to free-flow
    nmat[0, 0] = 2; nmat[0, 2] = 2; nmat[1, 0] = -2; nmat[1,1] = 5
    nmat[n_i, n_i] = 2; nmat[n_i, n_i - 2] = 2; nmat[n_i - 1, n_i] = -2; nmat[n_i - 1, n_i - 1] = 5

    # Node length
    global L_n
    L_n = L / n

    # Construct K matrix
    K = (E * I / L_n ** 3) * nmat

    # Construct the mass matrix.
    M = np.eye(n,n) * rho * A * L_n

    # Invert the matrix
    M_inv = np.linalg.inv(M)

    return K, M_inv
    
def coupling(pA: np.array, solver: Solver) -> np.array:
    # Extract state data
    cInd = (pA['label'] == ParticleType.Coupled)
    w = pA[cInd]['y']
    x = pA[cInd]['x']

    # Determine the pressure; using SPH interpolation
    pW = np.zeros_like(w)
    for i in range(len(w)):
        pW[i] = pressure_SPH(x[i], w[i], pA, solver)

    # Determine force on ice
    Fw = np.zeros_like(pW)

    # Compute accelerations
    a = np.matmul(M_inv, (pW + np.dot(K, w) + Fw))

    # Set the particle array
    pA['ay'][cInd] = a

    return pA

def pressure_SPH(x: float, y: float, pA: np.array, solver: Solver):
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
    h = 1.6 # Arbitrarily chosen; ish.

    if x < 0.0:
        return solver.method.rho0 * (y - y00)

    # Find the near particles
    h_i, _, dist, near_arr = solver.nn.nearPos(x, y, h, pA)

    # Compute kernel value
    w = solver.kernel.evaluate(dist, np.ones_like(h_i) * h)

    # Compute summation density at the location
    rho = SummationDensity(pA[near_arr]['label'], pA[near_arr]['m'], w)

    # Compute the pressure using TaitEOS, without -1.
    p = (rho / solver.method.rho0) ** solver.method.gamma * solver.method.B

    return p

def main():
    # ----- Setup ----- #
    # Simulation parameters
    duration = 3.0   # Duration of the simulation [s]
    height   = 10.0  # Height of the fluid box [m]
    width    = 10.0  # Width of the fluid box [m]

    # Computed parameters
    Nx = 40; ice_max = width / 2.0

    # Other parameters
    rho0 = 1000.0; XSPH = True 
    plot = True

    # Create some particles
    global y00
    y00, r0, pA = create_particles(Nx, rho0, width, height, ice_max)

    # -- Create matrix -- #

    # Beam/Ice properties
    L = ice_max * 3 # Total Length [m]
    b = 1       # Height of the beam [m]
    h = 1       # Width of the beam [m]

    # Compute properties
    n = len(pA[pA['label'] == ParticleType.Coupled]) # Number of Nodes [-]
    A = b * h                                        # Area [m^3]
    I = 1 / 12 * b * h ** 3                          # Area moment of Inertia [kg/m^3]

    # General properties
    v   = 0.3                  # Poisson ratio [-]
    # E   = 140e6 / (1 - v ** 2) # Youngs Modulus [Pa]
    E = 1
    rho = 916.0                # Density [kg/m^3]

    # Compute matrix
    global K, M_inv
    K, M_inv = createMatrix(n, E, I, A, rho, L)
    # coupling = None

    # Create the solver
    kernel     = Wendland()
    method     = WCSPH(height=height, rho0=rho0, r0=r0, useXSPH=XSPH, Pb=0, useSummationDensity=False)
    integrator = PEC(useXSPH=XSPH, strict=False)
    solver     = Solver(method, integrator, kernel, duration, quick=False, incrementalWriteout=False, maxSettle=0, exportProperties=['x', 'y', 'p', 'vx', 'vy'], coupling=coupling)

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
    exportPath = f'{sys.path[0]}/IceBreak.hdf5'
    solver.save(exportPath)

    if plot == True:
        plt = Plot(exportPath, title=f'IceBreak (2D); {len(pA)} particles', xmin=-ice_max, xmax=width + 2 * r0, ymin=- 2 * r0, ymax=height * 1.3 + 2 * r0)
        plt.save(f'{sys.path[0]}/IceBreak.mp4')

if __name__ == '__main__':
    main()

