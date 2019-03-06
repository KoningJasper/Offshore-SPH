from numba import njit, prange

@njit(fastmath=True)
def Continuity(p, comp):
    J = len(comp); _arho = 0.0
    for j in prange(J):
        # Manually compute the dot product.
        dot = comp[j]['vx'] * comp[j]['dw_x'] \
                + comp[j]['vy'] * comp[j]['dw_y']
        
        _arho += comp[j]['m'] * dot
    return _arho