# Lattice descriptors for numpy backend

from abc import ABC
from typing import TypeVar, Generic

ArrayImplementation = TypeVar('ArrayImplementation')

class LatticeDescriptor(ABC, Generic[ArrayImplementation]):
    """
    Abstract base class for all lattice descriptors
    """
    
    # The number of velocity directions
    Q: int

    # discretized velocity directions
    e: ArrayImplementation

    # weights for each direction
    w: ArrayImplementation

    # opposite directions-indices? for each direction
    opp: ArrayImplementation