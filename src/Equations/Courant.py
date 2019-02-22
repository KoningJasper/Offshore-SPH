from numba import njit

@njit(fastmath=True, cache=True)
def Courant(alpha, h, c):
    """
        Computes the courant condition.


        Parameters
        ----------

        alpha: Scaling parameter (generally 0.4)

        h: array of h's

        c: array of speeds of sound

        Returns
        -------

        float: min. time step according to courant condition.
    """

    J = len(h)
    assert(len(h) == len(c))

    h_min = 10e10
    c_max = 1e-10
    for j in range(J):
        if (h[j] < h_min):
            h_min = h[j]

        if (c[j] > c_max):
            c_max = c[j]

    # Compute courant
    return alpha * h_min / c_max