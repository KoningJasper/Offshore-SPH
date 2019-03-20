import numpy as np
from numba import njit

def Density(comp: np.array) -> float:
    """
        Summation density.

        Parameters
        ----------
        comp: np.array
            computed properties for neigbouring particles.

        Returns
        -------
        rho: float
            Density of the particle.
    """
    rho = 0.0
    for j in range(len(comp)):
        rho += comp[j]['m'] * comp[j]['w']
    return rho