import numpy as np

""" Common functions/helpers/objects for Offshore-SPH """

types = {
    'fluid': 0,
    'boundary': 1,
    'temp-boundary': 2
}

particle_dtype = np.dtype(
    {
        'names': ['type', 'm', 'rho', 'p', 'drho', 'h', 'x', 'y', 'vx', 'vy', 'ax', 'ay'],
        'types': [
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
        ]
    }
)

computed_dtype = np.dtype({
    'names': [
        # Single properties
        'h', 'q', 'c', 'r', 
        
        # Kernel
        'w',
        'dw_x',
        'dw_y',

        # Positional
        'x', 'y', 'vx', 'vy'
    ],
    'types': [
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
    ]
})