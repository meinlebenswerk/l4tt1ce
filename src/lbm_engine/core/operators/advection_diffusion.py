# ABCs for advection-diffusion operators

from abc import ABC, abstractmethod

# TODO -> we most likely want some typing for the operator functions
# but that only works if we know the underlying types of the implementation, aka what backend we are using
# so either we create a type-hint for generic ndarrays? which may work
# or we create a type hint for the backend we are using, or we make this generic and inject the backend
# in the implementation


class ADOperator(ABC):
    """
    Abstract base class for all advect-diff operators
    """

    @abstractmethod
    def __call__(self, g, u, phi) -> None:
        pass


class PulsedConcentrationDirichletOperator(ADOperator):
    pass

class ConstantScalarDirichletOperator(ADOperator):
    pass

class ZeroGradientOutletOperator(ADOperator):
    pass
