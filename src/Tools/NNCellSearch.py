import numpy as np
from typing import Tuple
from math import ceil, floor, sqrt
from numba import njit, prange, jit, jitclass
from numba import int64, float64, boolean
from src.Tools.NearNeighbours import NearNeighbours

spec = [
    ('pCells', int64[:, :]),
    ('cellCache', int64[:, :]),
    
    ('alpha', float64),
    ('strict', boolean),
    ('cells', int64),

    # Domain
    ('h', float64),
    ('xmin', float64),
    ('xmax', float64),
    ('ymin', float64),
    ('ymax', float64),
    ('xbins', int64),
    ('ybins', int64)
]
@jitclass(spec)
class NNCellSearch(NearNeighbours):
    def __init__(self, alpha: float, strict: bool):
        """
            Creates a near neighbourhood searcher, using a cell-search algorithm.

            Parameters
            ----------

            alpha: float
                scaling factor for box-size, default 2.0
        """
        self.alpha  = alpha
        self.strict = strict

    def update(self, pA: np.array):
        """
            Update the near neighbourhood search.

            Parameters
            ----------

            pA: np.array
                array of particles, of dtype src.Common.particle_dtype.
        """
        self._calcDomain(pA)
        self._assignCells(pA)
        
    def near(self, i: int, pA: np.array):
        """
            Find the neighbours near to a certain particle at index i.

            Parameters
            ----------

            i: int
                index of the particle to find neighbours off

            h: np.array

            Returns
            -------

            indexes: np.array
                List of indexes of near neighbours

        """

        # Find own cell
        own_x = self.pCells[i, 0]
        own_y = self.pCells[i, 1]

        # Calculate h factor, number of cells to take.
        h = pA[i]['h']
        h_fac = ceil(h / self.h)

        # Get list of neighbouring cells
        indexes = []
        for x_ in range(-h_fac, h_fac + 1, 1):
            x = own_x + x_
            if x < 0 or x > self.xbins:
                    continue

            for y_ in range(-h_fac, h_fac + 1, 1):
                y = own_y + y_

                if y < 0 or y > self.ybins:
                    continue

                # Get it
                i = self._flatIndex(x, y)
                indexes.extend(self.cellCache[i])
                
        # Clip -1
        indexes = np.array(indexes)
        pot = indexes[indexes > -1]

        if self.strict == True:
            # Compute distance to only keep actual
            act = []
            for p in pot:
                r  = sqrt((pA['x'][i] - pA['x'][p]) ** 2 + (pA['y'][i] - pA['y'][p]) ** 2)
                hi = 0.5 * (pA['h'][i] + pA['h'][p])
                q  = r / hi
                if q <= 3.0:
                    act.append(p)
            return np.array(act)
        else:
            return pot

    def _calcDomain(self, pA: np.array):
        """ Calculates the domain. """

        self.xmin = pA['x'].min()
        self.xmax = pA['x'].max()

        self.ymin = pA['y'].min()
        self.ymax = pA['y'].max()


        self.h = max(1, pA['h'].min())

        # Calculate the number of bins
        self.xbins = ceil((self.xmax - self.xmin) / (self.alpha * self.h))
        self.ybins = ceil((self.ymax - self.ymin) / (self.alpha * self.h))

        self.cells = self.xbins * max(1, self.ybins)

    def _index(self, x: float, y: float) -> Tuple[int, int]:
        """ Computes a indexes for a given (x, y) position. """
        xbin = floor(x / (self.alpha * self.h))
        ybin = floor(y / (self.alpha * self.h))
        return (xbin, ybin)

    def _flatIndex(self, x: int, y: int) -> int:
        return x + self.xbins * y

    def _assignCells(self, pA: np.array):
        """ Assigns cells to the particles based on their positions. """
        J = len(pA)
        self.pCells = np.zeros((J, 2), dtype=np.int64)
        cache       = np.full((self.cells, J), -1, dtype=np.int64)
        for j in prange(J):
            (x, y) = self._index(pA[j]['x'], pA[j]['y'])
            self.pCells[j][0] = x
            self.pCells[j][1] = y

            # Store in cache
            i  = self._flatIndex(x, y)
            ni = np.argmax(cache[i] == -1) # Get next index.
            cache[i][ni] = j

        # Convert to numpy
        self.cellCache = cache