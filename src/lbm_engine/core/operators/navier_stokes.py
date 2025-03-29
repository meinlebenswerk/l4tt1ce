# ABCs for navier-stokes operators

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

ArrayImplementation = TypeVar('ArrayImplementation')


class NSOperator(ABC, Generic[ArrayImplementation]):
    """
    Abstract base class for all navier stokes operators
    """

    @abstractmethod
    def __call__(self, f: ArrayImplementation, u: ArrayImplementation, rho: ArrayImplementation) -> tuple[ArrayImplementation, ArrayImplementation, ArrayImplementation]:
        pass


class BounceBackOperator(NSOperator):
    pass

class VelocityDirichletOperator(NSOperator):
    pass

class PressureDirichletOperator(NSOperator):
    pass
