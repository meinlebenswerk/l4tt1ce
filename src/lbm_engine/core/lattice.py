# ABC for lattice classes, implementations in impl/core/...

import numpy as np
from abc import ABC, abstractmethod
from src.lbm_engine.core.operators.advection_diffusion import ADOperator
from src.lbm_engine.core.operators.navier_stokes import NSOperator

from numpy.typing import NDArray

class Lattice2D(ABC):

    def __init__(self) -> None:
        self.operators: dict[str, NSOperator] = {}

    def addOperator(self, name: str, operator: NSOperator):
        self.operators[name] = operator

    @abstractmethod
    def step(self):
        """
        Implements the step function for this lattice with an optional number of steps
        """
        pass

    @property
    @abstractmethod
    def u_np(self) -> NDArray:
        pass

    @property
    @abstractmethod
    def rho_np(self) -> NDArray:
        pass
    
    def print(self):
        # --- Navier-Stokes Lattice ---
        print("Navier-Stokes Lattice".center(60))
        velocity_magnitude = np.sqrt(self.u_np[..., 0]**2 + self.u_np[..., 1]**2)
        avg_velocity = np.mean(velocity_magnitude)
        avg_rho = np.mean(self.rho_np)

        print(f"{'Average Velocity Magnitude (u)':<40}: {avg_velocity:10.3f}")
        print(f"{'Average Density (ρ)':<40}: {avg_rho:10.3f}")
    
class ScalarLattice2D(ABC):
    """
    Two-dimensional lattice but it contains a scalar field used for diffusive transport etc
    """

    def __init__(self) -> None:
        self.operators: dict[str, ADOperator] = {}

    def addOperator(self, name: str, operator: ADOperator):
        self.operators[name] = operator

    @property
    @abstractmethod
    def phi_np(self) -> NDArray:
        """
        Returns the scalar field (phi) of the lattice
        """
        pass

    @abstractmethod
    def step(self):
        """
        Implements the step function for this lattice with an optional number of steps
        """
        pass

    def print(self):
        # --- Advection-Diffusion Lattice ---
        print("-" * 60)
        print("Advection-Diffusion Lattice".center(60))
        avg_phi = np.mean(self.phi_np)
        min_phi = np.min(self.phi_np)
        max_phi = np.max(self.phi_np)

        print(f"{'Average Scalar Field (φ)':<40}: {avg_phi:10.3f}")
        print(f"{'Minimum Scalar Field (φₘᵢₙ)':<40}: {min_phi:10.3f}")
        print(f"{'Maximum Scalar Field (φₘₐₓ)':<40}: {max_phi:10.3f}")
