# ABCs for collision operators

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.lbm_engine.core.descriptor import LatticeDescriptor

# Allow the operators to be generic over the array implementation (numpy, jax, etc.)
ArrayImplementation = TypeVar('ArrayImplementation')

class CollisionOperator(ABC, Generic[ArrayImplementation]):
    """
    Abstract base class for collision operators
    """

    @abstractmethod
    def __init__(
        self,
        tau: float,
        descriptor: LatticeDescriptor[ArrayImplementation]
    ) -> None:
        pass

    @abstractmethod
    def compute_feq(
        self,
        rho: ArrayImplementation,
        u: ArrayImplementation,
        mask: ArrayImplementation | None = None
    ) -> ArrayImplementation:
        pass

    @abstractmethod
    def compute_delta_f(
        self,
        f_lattice: ArrayImplementation,
        phi: ArrayImplementation,
        u: ArrayImplementation,
        mask: ArrayImplementation | None = None
    ) -> ArrayImplementation:
        pass

class ADCollisionOperator(ABC, Generic[ArrayImplementation]):
    """
    Abstract base class for advection-diffusion collision operators
    """

    @abstractmethod
    def __init__(
        self,
        tau: float,
        descriptor: LatticeDescriptor[ArrayImplementation]
    ) -> None:
        pass

    # TODO: refactor out the descriptors from the sig, can be provided @init
    @abstractmethod
    def compute_feq(
        self,
        phi: ArrayImplementation,
        u: ArrayImplementation,
        mask: ArrayImplementation | None = None
    ) -> ArrayImplementation:
        pass

    # TODO: refactor out the descriptors from the sig, can be provided @init
    @abstractmethod
    def compute_delta_f(
        self,
        f_lattice: ArrayImplementation,
        phi: ArrayImplementation,
        u: ArrayImplementation,
        mask: ArrayImplementation | None = None
    ) -> ArrayImplementation:
        pass

class BGK_collisionOperator(CollisionOperator[ArrayImplementation]):
    pass

class BGK_AdvectionDiffusion_collisionOperator(ADCollisionOperator[ArrayImplementation]):
    pass