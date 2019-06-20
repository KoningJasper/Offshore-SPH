import numpy as np
from typing import List, Tuple
from numba import njit, jit, prange
from src.Common import computed_dtype, ParticleType
from src.Equations.SummationDensity import SummationDensity
from src.Equations.BoundaryForce import BoundaryForce

@njit(fastmath=True)
def computeProps(i: int, pA: np.array, near_arr: List[int], h_i, q_i, dist, evFunc, gradFunc):
    """
        Computes the computed properties (other particles).

        Parameters
        ----------

        i: int
            index of the active particle (self)

        pA: np.array
            particle array

        near_arr: np.array
            Array of indexes of particles which are near, calculated by the neigbourhood search.

        Returns
        -------

        calcProps: np.array
            Array of computed properties, with dtype: computed_dtype
    """
    # Just assign the easy props first
    calcProps = _assignProps(i, pA, near_arr, h_i, q_i, dist)

    # Kernel values
    w    = evFunc(calcProps['r'], calcProps['h'])
    dw_x = gradFunc(calcProps['x'], calcProps['r'], calcProps['h'])
    dw_y = gradFunc(calcProps['y'], calcProps['r'], calcProps['h'])

    for j in range(len(calcProps)):
        calcProps[j]['w']    = w[j] # Not needed for density change
        calcProps[j]['dw_x'] = dw_x[j]
        calcProps[j]['dw_y'] = dw_y[j]

    return calcProps

@jit(fastmath=True)
def findActive(J: int, pA: np.array) -> Tuple[float, np.array]:
    """
        Finds the active particles and returns their index and total active count.

        Parameters
        ----------

        J: int
            number of particles

        pA: np.array
            Particle array

        Returns
        -------

        active count: int
            Count of active particles

        active index: np.array
            Boolean array of the active particles

    """
    a_i = np.invert(np.array(pA['deleted'], copy=True, dtype=np.bool))
    a_c = np.sum(a_i)
    return a_c, a_i

@njit(fastmath=True)
def _assignProps(i: int, particleArray: np.array, near_arr: np.array, h_i: np.array, q_i: np.array, dist: np.array):
    J = len(near_arr)

    # Create empty array
    calcProps = np.zeros(J, dtype=computed_dtype)

    # Fill based on existing data.
    for j in range(J):
        global_i = near_arr[j]
        pA = particleArray[global_i]

        # From self properties
        calcProps[j]['label'] = pA['label']
        calcProps[j]['p']   = pA['p']
        calcProps[j]['m']   = pA['m']
        #calcProps[near_i]['c']   = self.method.compute_speed_of_sound(pA)
        calcProps[j]['rho'] = pA['rho']

        # Pre-calculated properties
        calcProps[j]['h'] = h_i[j] # average h, precalculated
        calcProps[j]['q'] = q_i[j] # dist / h, precalculated
        calcProps[j]['r'] = dist[j] # distance, precalculated

        # Positional values
        calcProps[j]['x']  = particleArray[i]['x'] - pA['x']
        calcProps[j]['y']  = particleArray[i]['y'] - pA['y']
        calcProps[j]['vx'] = particleArray[i]['vx'] - pA['vx']
        calcProps[j]['vy'] = particleArray[i]['vy'] - pA['vy']
    # END_LOOP
    return calcProps

@njit(fastmath=True, nogil=True)
def computeH(sigma: float, J: int, m: np.array, rho: np.array):
    """ Compute (dynamic) h size, based on Monaghan 2005. """
    d = 2 # 2 Dimensions (x, y)
    f = 1 / d
    h = np.zeros_like(m)
    
    # Outside of compute loop so prange can be used.
    for j in prange(J):
        if rho[j] > 1e-12:
            h[j] = sigma * (m[j] / rho[j]) ** f
    
    return h

@njit(fastmath=True, nogil=True)
def _loop(pA, evFunc, gradFunc, methodClass, nn):
    p = methodClass.compute_pressure(pA)
    c = methodClass.compute_speed_of_sound(pA)

    for i in prange(len(pA)):
        pA[i]['p'] = p[i]
        pA[i]['c'] = c[i]

    if (methodClass.useSummationDensity == True):
        # Extra summation loop
        for i in prange(len(pA)):
            if pA[i]['label'] != ParticleType.Fluid:
                continue

            # Find near neighbours and their h and q
            h_i, q_i, dist, near_arr = nn.near(i, pA)
            w = evFunc(dist, h_i)

            # Compute and set density
            pA[i]['rho'] = SummationDensity(pA[near_arr]['label'], pA[near_arr]['m'], w)

    # Regular loop
    for i in prange(len(pA)):
        if pA[i]['label'] != ParticleType.Fluid:
            continue

        # Find near neighbours and their h and q
        h_i, q_i, dist, near_arr = nn.near(i, pA)

        # Skip if got no neighbours, early exit.
        # Keep same properties, no acceleration.
        if len(near_arr) == 0:
            continue

        # Create computed properties
        # Fill the props
        calcProps = computeProps(i, pA, near_arr, h_i, q_i, dist, evFunc, gradFunc)

        # Continuity
        pA[i]['drho'] = methodClass.compute_density_change(pA[i], calcProps)

        # Momentum
        [a_x, a_y] = methodClass.compute_acceleration(pA[i], calcProps)
        
        # Compute boundary forces
        [b_x, b_y] = BoundaryForce(methodClass.r0, methodClass.D, methodClass.p1, methodClass.p2, pA[i], calcProps)

        # Total acceleration
        pA[i]['ax'] = a_x + b_x
        pA[i]['ay'] = a_y + b_y

        # XSPH
        [pA[i]['vx'], pA[i]['vy'], pA[i]['xsphx'], pA[i]['xsphy']] = methodClass.compute_velocity(pA[i], calcProps)

    return pA