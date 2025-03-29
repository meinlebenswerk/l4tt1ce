# Lattice descriptors for numpy backend

import jax
import jax.numpy as jnp

from src.lbm_engine.core.descriptor import LatticeDescriptor




class D2Q9Jax(LatticeDescriptor[jax.Array]):
    """
    D2Q9 Lattice Descriptor
    This is a 2D lattice with 9 velocity directions
    """
    Q = 9 #directions
    e = jnp.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=jnp.int32) #discretized velocity set
    w = jnp.array([4/9] + [1/9]*4 + [1/36]*4) #weights (is this correct? I hope so)
    opp = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=jnp.int32)
