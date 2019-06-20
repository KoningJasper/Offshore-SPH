import numba
import numpy as np
from src.Common import ParticleType

@numba.njit()
def Shepard(w: np.array, labels: np.array, m_b: np.array, rho_b: np.array):
    w_tilde = 0
    for i in range(len(w)):
        if rho_b[i] < 1e-3 or labels[i] != ParticleType.Fluid:
            continue
        w_tilde += w[i] * m_b[i] / rho_b[i]
    return w / w_tilde