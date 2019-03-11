import numpy as np
from math import sqrt
from numba import njit, vectorize

@vectorize('float64(float64, float64, float64, float64)', fastmath=True)
def TaitEOS(gamma, B, rho0, rho):
    """
        Calculates the pressure according to Tait EOS.

        Parameters
        ----------

        gamma: factor

        B: factor

        rho0: factor

        rho: array of densities

        Returns
        -------
        Array of pressures

    """
    ratio = (rho / rho0) ** gamma
    return (ratio - 1.0) * B

@njit(fastmath=True)
def TaitEOS_B(co, rho0, gamma):
    """
    Calculates the B parameter in the TaitEOS.
    """
    return co * co * rho0 / gamma

@njit(fastmath=True)
def TaitEOS_co(H):
    """ Calculates the speed of sound according to Tait EOS based on height of the water column (H). """
    # Monaghan (2002) p. 1746
    return 10.0 * sqrt(2 * 9.81 * H)

@vectorize('float64(float64, float64, float64, float64, float64)', fastmath=True)
def TaitEOS_height(rho0, H, B, gamma, y):
    """
        Initializes the particles with an initial pressure (P) based on their height (y) with relation to the water column height (H).
        
        Parameters
        ----------

        rho0: Initial density of the fluid.

        H: Water column height

        B: Tait parameter

        gamma: Tait exponent parameter

        y: Array of heights (y-coordinate).    
    """
    frac = rho0 * 9.81 * (H - y) / B
    return rho0 * (1 + frac) ** (1 / gamma)