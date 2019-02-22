# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest, os
import h5py
import numpy as np
import src.Common

class test_h5py_custom_export(unittest.TestCase):
    def test_small_dtype(self):
        data = np.zeros(10, dtype=[('mass', np.float64), ('label', np.int8)])
        with h5py.File('test.hdf5', 'w') as h5f:
            h5f.create_dataset('test', data=data)

        # Delete the file
        os.remove('test.hdf5')

    def test_custom_dtype(self):
        data = np.zeros(10, dtype=src.Common.particle_dtype)
        with h5py.File('test.hdf5', 'w') as h5f:
            h5f.create_dataset('test', data=data)

        # Delete the file
        os.remove('test.hdf5')

    def test_multi_dimensional_custom_dtype(self):
        data = []
        for i in range(10):
            data.append(np.zeros(10, dtype=src.Common.particle_dtype))

        with h5py.File('test.hdf5', 'w') as h5f:
            h5f.create_dataset('test', data=np.array(data))

        # Delete the file
        os.remove('test.hdf5')

if __name__ == "__main__":
    test_h5py_custom_export().test_small_dtype()
    test_h5py_custom_export().test_custom_dtype()
    test_h5py_custom_export().test_multi_dimensional_custom_dtype()
