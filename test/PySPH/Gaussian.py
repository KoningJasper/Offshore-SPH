import numpy as np
from math import exp


M_2_SQRTPI = 2.0 / np.sqrt(np.pi)
class Gaussian(object):
    r"""Gaussian Kernel: [Liu2010]_

    .. math::
             W(q) = \ &\sigma_g e^{-q^2}, \ & \textrm{for} \ 0\leq q \leq 3,\\
                  = \ & 0, & \textrm{for} \ q>3,\\

    where :math:`\sigma_g` is a dimensional normalizing factor for the gaussian
    function given by:

    .. math::
             \sigma_g  = \ & \frac{1}{\pi^{1/2} h}, \ & \textrm{for dim=1}, \\
             \sigma_g  = \ & \frac{1}{\pi h^2}, \ & \textrm{for dim=2}, \\
             \sigma_g  = \ & \frac{1}{\pi^{3/2} h^3}, & \textrm{for dim=3}. \\

    References
    ----------
    .. [Liu2010] `M. Liu, & G. Liu, Smoothed particle hydrodynamics (SPH):
        an overview and recent developments, "Archives of computational
        methods in engineering", 17.1 (2010), pp. 25-76.
        <http://link.springer.com/article/10.1007/s11831-010-9040-7>`_
    """
    fac: float; # Equal to 1 / pi, for 2D
    def __init__(self, dim=2):
        self.radius_scale = 3.0
        self.dim = dim

        self.fac = 0.5 * M_2_SQRTPI
        if dim > 1:
            self.fac *= 0.5 * M_2_SQRTPI
        if dim > 2:
            self.fac *= 0.5 * M_2_SQRTPI

    def get_deltap(self):
        # The inflection point is at q=1/sqrt(2)
        # the deltap values for some standard kernels
        # have been tabulated in sec 3.2 of
        # http://cfd.mace.manchester.ac.uk/sph/SPH_PhDs/2008/crespo_thesis.pdf
        return 0.70710678118654746

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
        h1 = 1. / h
        q = rij * h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        val = 0.0
        if (q < 3.0):
            val = exp(-q * q) * fac

        return val

    def dwdq(self, rij=1.0, h=1.0):
        h1 = 1. / h
        q = rij * h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        # compute the gradient
        val = 0.0
        if (q < 3.0):
            if (rij > 1e-12):
                val = -2.0 * q * exp(-q * q)

        return val * fac

    def gradient(self, xij=[0., 0., 0.], rij=1.0, h=1.0, grad=[0, 0, 0]):
        h1 = 1. / h

        # compute the gradient.
        if (rij > 1e-12):
            wdash = self.dwdq(rij, h)
            tmp = wdash * h1 / rij
        else:
            tmp = 0.0

        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        #grad[2] = tmp * xij[2]

    def gradient_h(self, xij=[0., 0., 0.], rij=1.0, h=1.0):
        h1 = 1. / h
        q = rij * h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        # kernel and gradient evaluated at q
        w = 0.0
        dw = 0.0
        if (q < 3.0):
            w = exp(-q * q)
            dw = -2.0 * q * w

        return -fac * h1 * (dw * q + w * self.dim)