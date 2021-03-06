import numpy as np
import enum
from collections import namedtuple
from numba import jit, njit

""" Common functions/helpers/objects for Offshore-SPH """

class ParticleType(enum.IntEnum):
    Fluid = 0,
    Boundary = 1,
    TempBoundary = 2,
    Coupled = 3

def get_label_code(label: str):
    if label == 'fluid':
        return ParticleType.Fluid
    elif label == 'boundary':
        return ParticleType.Boundary
    elif label == 'temp-boundary':
        return ParticleType.TempBoundary
    elif label == 'coupled':
        return ParticleType.Coupled
    else:
        raise Exception('Argument out of range')

particle_dtype = np.dtype(
    {
        'names': ['deleted', 'label', 'm', 'rho', 'p', 'c', 'drho', 'h', 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'xsphx', 'xsphy', 'x0', 'y0', 'vx0', 'vy0', 'rho0'],
        'formats': [
            np.bool,
            np.int8,
            np.double,
            np.double,
            np.double,
            np.double,
            np.double,
            np.double,
            np.double,
            np.double,
            np.double,
            np.double,
            np.double,
            np.double,

            # XSPH
            np.double,
            np.double,

            # Nought values.
            np.double,
            np.double,
            np.double,
            np.double,
            np.double,
        ]
    }
)

# Properties needed for computation
computed_dtype = np.dtype({
    'names': [
        'label',

        # Single properties
        'm', 'p', 'rho', 'h', 'q', 'c', 'r', 
        
        # Kernel
        'w',
        'dw_x',
        'dw_y',

        # Positional
        'x', 'y', 'vx', 'vy'
    ],
    'formats': [
        np.int8,
        
        # Single
        np.double,
        np.double,
        np.double,
        np.double,
        np.double,
        np.double,
        np.double,

        # Kernel
        np.double,
        np.double,
        np.double,

        # Positional
        np.double,
        np.double,
        np.double,
        np.double,
    ]
})

@njit(parallel=True, fastmath=True)
def _stack(m1, m2):
    """
        Stacks two arrays, horizontally.
    """
    return np.transpose(np.vstack((m1, m2)))