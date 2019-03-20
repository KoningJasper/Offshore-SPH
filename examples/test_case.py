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


def main():
    # ----- Setup ----- #
    # Simulation parameters
    duration = 10.0

    # Other parameters
    rho0 = 1000.0; XSPH = True 
    plot = True

    # Create some particles
    pA = np.zeros(2, dtype=particle_dtype)
    pA[1]['y'] = 1.0
    pA['m'] = 1.0
    pA['rho'] = rho0
    r0 = 1.0

    # Create the solver
    kernel     = CubicSpline()
    method     = WCSPH(height=1.0, rho0=rho0, r0=r0, useXSPH=XSPH)
    integrator = PEC(useXSPH=XSPH, strict=True)
    solver     = Solver(method, integrator, kernel, duration, quick=True, incrementalWriteout=False)

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
    exportPath = f'{sys.path[0]}\\test_case.hdf5'
    solver.save(exportPath)

    if plot == True:
        plt = Plot(exportPath, title=f'Ice Breaking (2D); {len(pA)} particles', xmin=-3, xmax=81, ymin=-3, ymax=41)
        plt.save(f'{sys.path[0]}\\test_case.mp4')

if __name__ == '__main__':
    main()

