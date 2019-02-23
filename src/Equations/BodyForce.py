import numpy as np
from numba import vectorize

@vectorize('float64(float64, float64)', fastmath=True, cache=True)
def BodyForce(f, a):
    """
        Adds a body force/acceleration to the acceleration.

        Parameters
        ----------

        f: float, force/acceleration to add to the 
        
        a: array, acceleration of the particles

        Returns
        -------

        array of acceleration
    """

    a += f
    return a
