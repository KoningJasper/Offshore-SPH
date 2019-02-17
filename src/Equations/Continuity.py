from numba import njit, prange

@njit(parallel=True, fastmath=True)
def Continuity(m, dwij, vij):
    J = len(m); _arho = 0.0
    for j in prange(J):
        # Manually compute the dot product.
        dot = vij[j, 0] * dwij[j, 0] + vij[j, 1] * dwij[j, 1]
        
        _arho += m[j] * dot
    return _arho