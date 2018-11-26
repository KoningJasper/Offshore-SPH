import numpy as np
import scipy.spatial

from src.Config.Props import Props
from src.Mesh.Uniform import Uniform


class Main:
    """ MAIN: This function starts the SPH program. """
    props: Props
    coords: np.array

    def __init__(self):

        # Build the config
        self.props = Props(nodes=100)

        # Create a uniform mesh-grid
        self.coords = Uniform.create(self.props)

        # Calculate the distance
        dist = scipy.spatial.distance.cdist(coords, coords, 'euclidean')


if __name__ == '__main__':
    Main()
