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
        'formats': [
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
            np.double
        ]
    }
)

# Properties needed for computation
computed_dtype = np.dtype({
    'names': [
        # Single properties
        'm', 'rho', 'h', 'q', 'c', 'r', 
        
        # Kernel
        'w',
        'dw_x',
        'dw_y',

        # Positional
        'x', 'y', 'vx', 'vy'
    ],
    'formats': [
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
        np.double,
    ]
})