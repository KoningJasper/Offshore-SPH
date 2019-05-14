import numpy as np
from numba import njit
from src.Common import ParticleType

@njit(fastmath=True)
def SummationDensity(labels: np.array, m: np.array, w: np.array) -> float:
    rho = 0.0
    for j in range(len(labels)):
        if labels[j] != ParticleType.Fluid:
            continue
        else:
            rho += m[j] * w[j]
    return rho