# Add parent folder to path
import sys, os, math
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from src.Solver import Solver
from src.Helpers import Helpers
from src.Methods.WCSPH import WCSPH
from src.Kernels.CubicSpline import CubicSpline
from src.Integrators.PEC import PEC
from src.Post.Plot import Plot
from src.Common import ParticleType, particle_dtype

def create_particles(Nx: int, rho0: float, x_end: float, y_end: float):
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
    x_min = - 2 * r0
    x_max = x_end + 2 * r0
    y_min = - 2 * r0
    y_max = y_end + 2 * r0 + 2 * y_end
    mass_b = mass * 1.0 # Mass of the boundary [kg]
    
    # Create the boundary
    bottom = Helpers.rect(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_min, r0=r0, mass=mass_b, rho0=rho0, label=ParticleType.Boundary)
    left   = Helpers.rect(xmin=x_min, xmax=x_min, ymin=y_min, ymax=y_max, r0=r0, mass=mass_b, rho0=rho0, label=ParticleType.Boundary)
    right  = Helpers.rect(xmin=x_max, xmax=x_max, ymin=y_min, ymax=y_max, r0=r0, mass=mass_b, rho0=rho0, label=ParticleType.Boundary)

    # Ice and plate.
    #p = plate(r0, x_end - 20, y_end + 10, mass_b, rho0)
    #i = ice(r0, y_max + r0, mass_b, rho0, 10.0)

    return r0, np.concatenate((fluid, bottom, left, right))

def plate(r0: float, x: float, y: float, m: float, rho0: float):
    l = 25; angle = 45.0
    N = math.ceil(l / r0)
    plate = np.zeros(N, dtype=particle_dtype)
    plate['label'] = ParticleType.Boundary
    plate['m']     = m
    plate['rho']   = rho0

    # Rotate with $angle$ deg.
    plate['x'] = x + np.linspace(0, l, N) * np.cos(np.deg2rad(angle))
    plate['y'] = y - np.linspace(0, l, N) * np.sin(np.deg2rad(angle))

    return plate

def ice(r0: float, y: float, m: float, rho0: float, vx: float):
    l = 100; N = math.ceil(l / r0); x_start = -15
    ice = np.zeros(N, dtype=particle_dtype)
    ice['label'] = ParticleType.Boundary
    ice['vx']    = vx; ice['xsphx'] = vx
    ice['m']     = m
    ice['rho']   = rho0
    ice['y']     = y
    ice['x']     = np.linspace(x_start, x_start + l, N)

    # Copy 3 times
    i1 = np.array(ice, copy=True)
    i2 = np.array(ice, copy=True)
    i3 = np.array(ice, copy=True)
    i2['y'] = y + r0; i3['y'] = y + 2 * r0

    return np.concatenate((i1, i2, i3))

def main():
    # ----- Setup ----- #
    # Simulation parameters
    Nx = 30; duration = 5.0
    height = 1.0; width = 1.0

    # Other parameters
    rho0 = 1000.0; XSPH = False 
    plot = True

    # Create some particles
    r0, pA = create_particles(Nx, rho0, width, height)

    # Create the solver
    kernel     = CubicSpline()
    method     = WCSPH(height=height, rho0=rho0, r0=r0, useXSPH=XSPH)
    integrator = PEC(useXSPH=XSPH, strict=False)
    solver     = Solver(method, integrator, kernel, duration, quick=False, incrementalWriteout=False, maxSettle=25_000, kE=0.0)

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
    exportPath = f'{sys.path[0]}\\ice-chris.hdf5'
    solver.save(exportPath)

    if plot == True:
        plt = Plot(exportPath, title=f'Ice Breaking (2D); {len(pA)} particles', xmin=-0.5, xmax=1.5, ymin=-0.5, ymax=1.5)
        plt.save(f'{sys.path[0]}\\ice-chris.mp4')

if __name__ == '__main__':
    main()

