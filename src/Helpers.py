import numpy as np
from math import ceil
from src.Common import particle_dtype, ParticleType

class Helpers():
    def __init__(self, rho0: float = 1000.):
        """ This class contains some particle helpers. """
        self.rho0 = rho0

    def box(self, xmin: float, xmax: float, ymin: float, ymax: float, N: int = None, r0: float = None, hex: bool = True, type: ParticleType = ParticleType.Fluid) -> np.array:
        """
            Creates a particle array that represents a box in a square configuration.

            Parameters
            ----------
            xmin: float
                Start of the box
            xmax: float
                End of the box
            ymin: float
                Start of the box in y-direction
            ymax: float
                End of the box in y-direction
            N: int
                Number of particles to use for the box, in-total. Either r0 or N needs to be set.
            r0: float
                Separation between particles. Either r0 or N needs to be set.
            hex: bool
                Should the particles be hex-packed, more energy efficient. Reduces settling time; generally recommended.
            type: ParticleType
                The particle type of the particle
            
            Returns
            -------
            particles: np.array
                Particle array of type particle_dtype with x and y filled in with the box values.

            Notes
            -----
            The number of actual particles can deviate from the set target if the box is not square and or if hex-packing is enabled.
        """

        # Check condition that atleast one of the two is set.
        assert((N != None) or (r0 != None))

        # Determine particle separation
        xrange = xmax - xmin
        yrange = ymax - ymin

        if N != None:
            N2 = np.sqrt(N)
            Nx = max(1, ceil(xrange / yrange * N2))
            Ny = max(1, ceil(yrange / xrange * N2))
        else:
            Nx = max(1, ceil(xrange / r0))
            Ny = max(1, ceil(yrange / r0))

        # Create the mesh-grid
        xv = np.linspace(xmin, xmax, Nx)
        yv = np.linspace(ymin, ymax, Ny)
        x, y = np.meshgrid(xv, yv)

        if hex == True:
            # Shift particles by half particle separation for hexgrid
            x[1::2] = x[1::2] + xrange / (Nx - 1) / 2
            x = x.ravel(); y = y.ravel()

            # Delete the ones outside the domain.
            mask = np.ones(len(x), dtype=np.bool)
            mask[2*Nx-1::2*Nx] = False
            x = x[mask]; y = y[mask]
        else:
            x = x.ravel(); y = y.ravel()

        # Assign to a particle array
        pA = np.zeros(len(x), dtype=particle_dtype)
        pA['x'] = x; pA['y'] = y;
        
        # Initialize other fields
        pA['label'] = type
        pA['rho']   = self.rho0

        return pA

        



