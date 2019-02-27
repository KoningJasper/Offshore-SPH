import numpy as np
from math import floor, ceil
from typing import List

class NNLinkedList():
    heads: np.array
    nexts: np.array
    shifts: np.array

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    cell_size: float
    scale: float
    n_cells: int
    cells_per_dim: List[int]

    def __init__(self):
        self.scale = 2.0
        self.shifts = np.array([-1, 0, 1])

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

        if ncx < 0 or ncy < 0:
            msg = 'LinkedListNNPS: Number of cells is negative (%s, %s).' % (ncx, ncy)
            raise RuntimeError(msg)

        ncx = 1 if ncx == 0 else ncx # max(1, ncx)
        ncy = 1 if ncy == 0 else ncy

        # number of cells along each coordinate direction
        self.ncells_per_dim = [ncx, ncy]

        # total number of cells
        _ncells = ncx * ncy
        return _ncells

    def bin(self, pA: np.array):
        """ Bin the particles. """
        for i in range(len(pA)):
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

        # Search radius
        hi2 = (self.scale * pA[i]['h']) ** 2
        
        # Search through neighbouring cells
        nbrs = []
        for ix in range(3):
            for iy in range(3):
                cid_x = _cid_x + self.shifts[ix]
                cid_y = _cid_y + self.shifts[iy]

                # Get cell index
                cell_index = self.get_valid_cell_index(cid_x, cid_y, self.cells_per_dim, self.n_cells)

                if cell_index > -1:
                    # get the first particle and begin iteration
                    _next = self.heads[ cell_index ]
                    while( _next != -1 ):
                        hj2 = (self.scale * pA[_next]['h']) ** 2
                        xij2 = self.norm2(pA[_next]['x'] - x, pA[_next]['y'] - y)

                        # select neighbor
                        if ( (xij2 < hi2) or (xij2 < hj2) ):
                            nbrs.append(_next)

                        # get the 'next' particle in this cell
                        _next = self.nexts[_next]

        return nbrs

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
