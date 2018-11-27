class Euler:
    """ A simplistic Euler solver. """

    stepsize: float  # The timestep used

    def __init__(self, stepsize: float):
        self.stepsize = stepsize

    @classmethod
    def integrate(self, state, t_n: float):
        """ Integrates the state at timestep (t_n) to timestep t_n + stepsize """
        return None
