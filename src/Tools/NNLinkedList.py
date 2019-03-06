import numpy as np
from math import floor, ceil, sqrt
from numba import jitclass, float64, boolean, int64
from typing import List


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
]
@jitclass(spec)
class NNLinkedList:
    def __init__(self, scale: float):
        """
            Initialize a new Linked List Neighbourhood finder. Mostly take from PySPH, modified slightly for numba.

            Parameters
            ---------
            scale: float
                Scaling factor for the box size, normally 2.0
        """
        self.scale  = scale

        # Create the shifts array
        self.shifts = np.array([-1, 0, 1], dtype=np.int64)

    def update(self, pA: np.array):
        self._init(pA)
        self._bin(pA)

    def near(self, i: int, pA: np.array):
        x = pA[i]['x']; y = pA[i]['y']; h = pA[i]['h']

        # Find unflattened id of particle
        _cid_x, _cid_y = self._find_cell_id_raw(x - self.xmin, y - self.ymin, self.cell_size)
        
        # Search through neighbouring cells
        nbrs = []
        q_i = []
        h_i = []
        r_i = []
        for ix in range(3):
            for iy in range(3):
                cid_x = _cid_x + self.shifts[ix]
                cid_y = _cid_y + self.shifts[iy]

                # Get cell index
                cell_index = self._get_valid_cell_index(cid_x, cid_y, self.ncells_per_dim, self.n_cells)

                if cell_index > -1:
                    # get the first particle and begin iteration
                    _next = self.heads[ cell_index ]
                    while( _next != -1 ):
                        # Compute the distance
                        xij2 = self._norm2(x - pA[_next]['x'], y - pA[_next]['y'])
                        r    = sqrt(xij2)

                        # Compute q = r/h
                        h_ij = 0.5 * (h + pA[_next]['h'])
                        q_ij = r / h_ij

                        # select neighbour
                        if (q_ij <= 3.0):
                            nbrs.append(_next)
                            r_i.append(r)
                            q_i.append(q_ij)
                            h_i.append(h_ij)

                        # get the 'next' particle in this cell
                        _next = self.nexts[_next]

        return np.array(h_i, dtype=np.float64), np.array(q_i, dtype=np.float64), np.array(r_i, dtype=np.float64), np.array(nbrs, dtype=np.uint64)

    def _init(self, pA: np.array):
        """ Initialize the arrays. """
        # Find mins and maxes
        self.xmin = pA['x'].min()
        self.xmax = pA['x'].max()

        self.ymin = pA['y'].min()
        self.ymax = pA['y'].max()

        # Compute grid
        self.cell_size = self._get_cell_size(pA)
        self.n_cells   = self._get_number_of_cells(pA)

        # Create the arrays
        self.heads = np.full(self.n_cells, -1, dtype=np.int64)
        self.nexts = np.full(len(pA), -1, dtype=np.int64)

    def _get_cell_size(self, pA: np.array):
        hmin      = pA['h'].min()
        cell_size = hmin * self.scale

        if cell_size < 1e-6:
            cell_size = 1.0

        return cell_size

    def _get_number_of_cells(self, pA: np.array):
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

    def _bin(self, pA: np.array):
        """ Bin the particles. """
        for i in range(len(pA)):
            x = pA[i]['x'] - self.xmin
            y = pA[i]['y'] - self.ymin

            # Get the cell-id
            _cid_x, _cid_y = self._find_cell_id_raw(x, y, self.cell_size)
            _cid = self._flatten_raw(_cid_x, _cid_y, self.ncells_per_dim)

            # Insert
            self.nexts[i]    = self.heads[_cid]
            self.heads[_cid] = i

    def _norm2(self, x, y):
        return x * x + y * y

    def _find_cell_id_raw(self, x: float, y: float, cell_size: float) -> (int, int):
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
        return self._real_to_int(x, cell_size), self._real_to_int(y, cell_size)

    def _real_to_int(self, val: float, step: float) -> int:
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

    def _get_valid_cell_index(self, cid_x: int, cid_y: int, ncells_per_dim: List[int], n_cells: int) -> int:
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
            cell_index = self._flatten_raw(cid_x, cid_y, ncells_per_dim)

            if not (-1 < cell_index < n_cells):
                cell_index = -1

        return cell_index

    def _flatten_raw(self, x: int, y: int, ncells_per_dim: List[int]) -> int:
        """Return a flattened index for a cell
        The flattening is determined using the row-order indexing commonly
        employed in SPH. This would need to be changed for hash functions
        based on alternate orderings.
        """
        ncx = ncells_per_dim[0]

        return x + ncx * y
