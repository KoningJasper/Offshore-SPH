import numpy as np


class Utils:
    """ Utility class for editing meshes. """

    @staticmethod
    def split(coords: np.array) -> (np.array, np.array):
        """ Return the coordinates split up into X and Y arrays. """
        return np.hsplit(coords, 2)
