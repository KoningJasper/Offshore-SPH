from dataclasses import dataclass


@dataclass()
class Props:
    """ Number of nodes. """
    N: int

    """ Length of axis, equal to sqrt(N). """
    L: int

    """ Particle mass, based on density and volume. """
    m: float

    rho: float

    def __init__(self, nodes: int, density: float, volume: float):
        self.N = nodes
        self.rho = density

        self.L = self.N ** 0.5
        self.m = (volume / nodes) * density
