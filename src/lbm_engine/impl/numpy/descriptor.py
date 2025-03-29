# Lattice descriptors for jax backend

import numpy as np
from numpy.typing import NDArray
from src.lbm_engine.core.descriptor import LatticeDescriptor




class D2Q9Numpy(LatticeDescriptor[NDArray]):
    """
    D2Q9 Lattice Descriptor
    This is a 2D lattice with 9 velocity directions
    """
    Q = 9 #directions
    e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32) #discretized velocity set
    w = np.array([4/9] + [1/9]*4 + [1/36]*4) # weights (is this correct? I hope so)
    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
