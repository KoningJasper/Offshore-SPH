import numpy as np


class Continuity():
    def Continuity(self, mass: float, dwij: np.array, vij: np.array) -> float:
        """
            SPH continuity equation; Calculates the change is density of the particles.
        """
        # Init
        _arho = 0.0

        # Calc change in density
        vijdotwij = np.sum(vij * dwij, axis=1) # row by row dot product
        _arho = np.sum(mass * vijdotwij, axis=0)

        return _arho
