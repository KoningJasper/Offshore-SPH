import numpy as np
import math
from numba import njit, prange
from src.Common import ParticleType

@njit(fastmath=True)
def BoundaryForce(r0, D, p1, p2, p, comp):
    """
        Computes the boundary force based on the equations provided in Monaghan 1992

        Parameters
        ----------

        r0: Initial spacing between particles

        D: Force factor of dimension [v * v], for dams/bores/weirs equal to 5gH.

        p1: Exponent one, equal to 4

        p2: Exponent two, equal to 2

        p: Particle array

        comp: Computed particle array

        Returns
        -------
    """
    f = [0., 0.]
    J = len(comp)
    for j in prange(J):
        if (comp[j]['label'] != ParticleType.Boundary) or (comp[j]['r'] > r0):
            # Only consider boundaries and where distance is closer than r0.
            continue
        elif comp[j]['r'] > 1e-12:
            frac = r0 / comp[j]['r']
            tmp = math.pow(frac, p1) - math.pow(frac, p2)
            fac = D * tmp

            f[0] += fac * comp[j]['x'] / math.pow(comp[j]['r'], 2)
            f[1] += fac * comp[j]['y'] / math.pow(comp[j]['r'], 2)
    return f
