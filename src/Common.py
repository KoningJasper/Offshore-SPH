import numpy as np

""" Common functions/helpers/objects for Offshore-SPH """

types = {
    'fluid': 0,
    'boundary': 1,
    'temp-boundary': 2
}

def get_label_code(label: str) -> int:
    return types[label]

def get_label(code: int) -> str:
    # TODO: Make this work.
    return ''

particle_dtype = np.dtype(
    {
        'names': ['label', 'm', 'rho', 'p', 'c', 'drho', 'h', 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'x0', 'y0', 'vx0', 'vy0', 'rho0'],
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