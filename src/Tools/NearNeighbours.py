import abc
import numpy as np

class NearNeighbours():
    @abc.abstractmethod
    def update(self, pA: np.array):
        """
            Update the near neighbourhood search.

            Parameters
            ----------

            pA: np.array
                array of particles, of dtype src.Common.particle_dtype.
        """
        pass

    @abc.abstractmethod
    def near(self, i: int, pA: np.array):
        """
            Find the neighbours near to a certain particle at index i.

            Parameters
            ----------

            i: int
                index of the particle to find neighbours off

            Returns
            -------

            indexes: np.array
                List of indexes of near neighbours

        """
        pass