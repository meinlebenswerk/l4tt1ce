# ABCs for navier-stokes operators

from abc import ABC, abstractmethod

# TODO -> we most likely want some typing for the operator functions
# but that only works if we know the underlying types of the implementation, aka what backend we are using
# so either we create a type-hint for generic ndarrays? which may work
# or we create a type hint for the backend we are using, or we make this generic and inject the backend
# in the implementation


class NSOperator(ABC):
    """
    Abstract base class for all navier stokes operators
    """

    @abstractmethod
    def __call__(self, f, u, rho) -> None:
        pass


class BounceBackOperator(NSOperator):
    pass

class VelocityDirichletOperator(NSOperator):
    pass

class PressureDirichletOperator(NSOperator):
    pass
