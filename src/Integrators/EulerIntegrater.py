from src.Particle import Particle


class EulerIntegrater():
    """ Stupidly simple Euler Integrator """

    @classmethod
    def integrate(self, dt: float, p: Particle, XSPH: bool = False):
        """
        Parameters:
        dt: time-step
        p: Particle to integrate
        """

        # Only move fluid particles.
        if p.label == 'fluid':
            p.v = p.v + dt * p.a

            if XSPH == True:
                p.r = p.r + dt * p.v + dt * p.vx
            else:
                p.r = p.r + dt * p.v

        p.rho = p.rho + dt * p.drho