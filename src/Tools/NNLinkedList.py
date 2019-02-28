import numpy as np
from math import floor, ceil, sqrt
from numba import jitclass, float64, int64, prange, boolean
from typing import List
from src.Tools.NearNeighbours import NearNeighbours


spec = [
    ('heads', int64[:]),
    ('nexts', int64[:]),
    ('shifts', int64[:]),

    ('xmin', float64),
    ('xmax', float64),
    ('ymin', float64),
    ('ymax', float64),
    ('cell_size', float64),
    ('scale', float64),
    ('n_cells', int64),
    ('ncells_per_dim', int64[:]),
    ('strict', boolean),
]
@jitclass(spec)
class NNLinkedList(NearNeighbours):
    def __init__(self, scale: float, strict: bool):
        """
            Initialize a new Linked List Neighbourhood finder.

            Parameters
            ---------
            scale: float
                Scaling factor for the box size, normally 2.0
            strict: bool
                Toggles distance checking.
        """
        self.scale  = scale
        self.strict = strict

        # Create the shifts array
        self.shifts = np.array([-1, 0, 1], dtype=np.int64)

    def update(self, pA: np.array):
        self.init(pA)
        self.bin(pA)

    def init(self, pA: np.array):
        """ Initialize the arrays. """
        # Find mins and maxes
        self.xmin = pA['x'].min()
        self.xmax = pA['x'].max()

        self.ymin = pA['y'].min()
        self.ymax = pA['y'].max()

        # Compute grid
        self.cell_size = self.get_cell_size(pA)
        self.n_cells   = self.get_number_of_cells(pA)

        # Create the arrays
        self.heads = np.full(self.n_cells, -1, dtype=np.int64)
        self.nexts = np.full(len(pA), -1, dtype=np.int64)

    def get_cell_size(self, pA: np.array):
        # TODO: Use looped max
        hmax = pA['h'].max()

        cell_size = hmax * self.scale

        if cell_size < 1e-6:
            cell_size = 1.0

        return cell_size

    def get_number_of_cells(self, pA: np.array):
        cell_size1 = 1./ self.cell_size

        # calculate the number of cells.
        ncx = ceil( cell_size1 * (self.xmax - self.xmin) )
        ncy = ceil( cell_size1 * (self.ymax - self.ymin) )

        # Ensure a minimum of 1.
        ncx = max(1, ncx)
        ncy = max(1, ncy)

        # number of cells along each coordinate direction
        self.ncells_per_dim = np.array([ncx, ncy], dtype=np.int64)

        # total number of cells
        return ncx * ncy

    def bin(self, pA: np.array):
        """ Bin the particles. """
        for i in prange(len(pA)):
            x = pA[i]['x'] - self.xmin
            y = pA[i]['y'] - self.ymin

            # Get the cell-id
            _cid_x, _cid_y = self.find_cell_id_raw(x, y, self.cell_size)
            _cid = self.flatten_raw(_cid_x, _cid_y, self.ncells_per_dim)

            # Insert
            self.nexts[i]    = self.heads[_cid]
            self.heads[_cid] = i

    def near(self, i: int, pA: np.array):
        x = pA[i]['x']; y = pA[i]['y']

        # Find unflattened id of particle
        _cid_x, _cid_y = self.find_cell_id_raw(x - self.xmin, y - self.ymin, self.cell_size)
        
        # Search through neighbouring cells
        nbrs = []
        for ix in range(3):
            for iy in range(3):
                cid_x = _cid_x + self.shifts[ix]
                cid_y = _cid_y + self.shifts[iy]

                # Get cell index
                cell_index = self.get_valid_cell_index(cid_x, cid_y, self.ncells_per_dim, self.n_cells)

                if cell_index > -1:
                    # get the first particle and begin iteration
                    _next = self.heads[ cell_index ]
                    while( _next != -1 ):
                        # Should the distance be checked?
                        if self.strict == True:
                            # Compute the distance
                            xij2 = self.norm2(pA[_next]['x'] - x, pA[_next]['y'] - y)
                            r    = sqrt(xij2)

                            # Compute q = r/h
                            h_ij = 0.5 * (pA[_next]['h'] + pA[i]['h'])
                            q_ij = r / h_ij

                            # select neighbour
                            if (q_ij <= 3.0):
                                nbrs.append(_next)
                        else:
                            nbrs.append(_next)

                        # get the 'next' particle in this cell
                        _next = self.nexts[_next]

        # Pad the array to pA length.
        arr = np.full(len(pA), -1, dtype=np.int64)
        arr[0:len(nbrs)] = np.array(nbrs)
        return arr

    def norm2(self, x, y):
        return x * x + y * y

    def find_cell_id_raw(self, x: float, y: float, cell_size: float) -> (int, int):
        """
        Find the cell index for the corresponding point
        
        Parameters
        ----------
        x, y: double
            the point for which the index is sought
        cell_size : double
            the cell size to use
        
        Notes
        ------
        Performs a box sort based on the point and cell size
        Uses the function  `real_to_int`
        """
        return self.real_to_int(x, cell_size), self.real_to_int(y, cell_size)

    def real_to_int(self, val: float, step: float) -> int:
        """
        Return the bin index to which the given position belongs.
        
        Parameters
        ----------
        val: float
            The coordinate location to bin
        step: float
            the bin size
        """

        return floor(val / step)

    def get_valid_cell_index(self, cid_x: int, cid_y: int, ncells_per_dim: List[int], n_cells: int) -> int:
        """Return the flattened index for a valid cell"""
        ncx = ncells_per_dim[0]
        ncy = ncells_per_dim[1]

        cell_index = -1

        # basic test for valid indices. Since we bin the particles with
        # respect to the origin, negative indices can never occur.
        is_valid: bool = (
            (ncx > cid_x > -1) and (ncy > cid_y > -1)
        )

        # Given the validity of the cells, return the flattened cell index
        if is_valid:
            cell_index = self.flatten_raw(cid_x, cid_y, ncells_per_dim)

            if not (-1 < cell_index < n_cells):
                cell_index = -1

        return cell_index

    def flatten_raw(self, x: int, y: int, ncells_per_dim: List[int]) -> int:
        """Return a flattened index for a cell
        The flattening is determined using the row-order indexing commonly
        employed in SPH. This would need to be changed for hash functions
        based on alternate orderings.
        """
        ncx = ncells_per_dim[0]

        return x + ncx * y
