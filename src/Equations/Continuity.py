from numba import njit
from src.Common import ParticleType

@njit(fastmath=True)
def Continuity(p, comp):
    J = len(comp); _arho = 0.0
    for j in range(J):
        # Only do fluid to fluid
        if comp[j]['label'] != ParticleType.Fluid:
            continue

        # Manually compute the dot product.
        dot = comp[j]['vx'] * comp[j]['dw_x'] \
                + comp[j]['vy'] * comp[j]['dw_y']
        
        _arho += comp[j]['m'] * dot
    return _arho