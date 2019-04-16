from numba import njit, prange
from typing import List
from src.Common import ParticleType

@njit(fastmath=True)
def Momentum(alpha, beta, p, comp) -> List[float]:
    """
        Monaghan Momentum equation


        Parameters:
        ------

        alpha: Scaling parameter for artifical viscosity.

        beta: Scaling parameter for artifical viscosity.

        p: Self-particle, [particle_dtype]

        comp: Other particles array

        Returns:
        ------

        acceleration: [ax, ay]
    """

    slf = p['p'] / (p['rho'] * p['rho']) # Lifted from the loop since it's constant.

    a = [0.0, 0.0]
    J = len(comp)
    for j in range(J):
        if comp[j]['label'] != ParticleType.Fluid:
            continue

        # Compute acceleration due to pressure.
        othr = comp[j]['p'] / (comp[j]['rho'] * comp[j]['rho'])

        # (Artificial) Viscosity
        dot = comp[j]['vx'] * comp[j]['x'] + comp[j]['vy'] * comp[j]['y']
        PI_ij = 0.0
        if dot < 0:
            # Averaged properties
            hij = 0.5 * (p['h'] + comp[j]['h']) # Averaged h.
            cij = 0.5 * (p['c'] + comp[j]['c']) # Averaged speed of sound.
            rhoij = 0.5 * (p['rho'] + comp[j]['rho']) # Averaged density.

            muij = hij * dot / (comp[j]['r'] * comp[j]['r'] + 0.01 * hij * hij)
            ppij = muij * (beta * muij - alpha * cij)
            PI_ij = ppij / rhoij
        
        # Compute final factor
        factor = slf + othr + PI_ij

        # Compute acceleration
        a[0] += - comp[j]['m'] * factor * comp[j]['dw_x']
        a[1] += - comp[j]['m'] * factor * comp[j]['dw_y']
    return a
