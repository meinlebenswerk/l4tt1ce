# Numpy implementations of operators

import numpy as np
from numpy.typing import NDArray

from src.lbm_engine.core.operators.navier_stokes import (
    BounceBackOperator,
    VelocityDirichletOperator,
    PressureDirichletOperator
)

from src.lbm_engine.core.operators.advection_diffusion import (
    PulsedConcentrationDirichletOperator,
    ConstantScalarDirichletOperator,
    ZeroGradientOutletOperator
)

from src.lbm_engine.core.operators.collision import (
    BGK_collisionOperator,
    BGK_AdvectionDiffusion_collisionOperator
)

from src.lbm_engine.core.descriptor import LatticeDescriptor

# Navier-stokes operators

class BounceBackOperatorJax(BounceBackOperator):
    def __init__(self, descriptor, mask):
        self.opp = descriptor.opp
        self.mask = mask

    def __call__(self, f, u=None, rho=None):
        mask = np.transpose(self.mask) if self.mask.shape != f.shape[:2] else self.mask
        for i in range(len(self.opp)):
            f[mask, i] = f[mask, self.opp[i]]


class VelocityDirichletOperatorJax(VelocityDirichletOperator):
    def __init__(self, descriptor, collisionOperator, mask, velocity_func):
        self.descriptor = descriptor
        self.collisionOperator = collisionOperator
        self.e = descriptor.e
        self.w = descriptor.w
        self.Q = descriptor.Q
        self.mask = mask
        self.velocity_func = velocity_func

    def __call__(self, f, u, rho):
        mask = np.transpose(self.mask) if self.mask.shape != f.shape[:2] else self.mask
        u[mask] = self.velocity_func(u[mask].shape)
        rho[mask] = 1.0  # assume constant pressure
        u2 = u[:,:,0]**2 + u[:,:,1]**2
        feq = self.collisionOperator.compute_feq(rho, u, mask)
        f[mask] = feq[mask]


class PressureDirichletOperatorJax(PressureDirichletOperator):
    def __init__(self, descriptor, collisionOperator, mask, rho_value):
        self.descriptor = descriptor
        self.collisionOperator = collisionOperator
        self.e = descriptor.e
        self.w = descriptor.w
        self.Q = descriptor.Q
        self.mask = mask
        self.rho_value = rho_value

    def __call__(self, f, u, rho):
        mask = np.transpose(self.mask) if self.mask.shape != f.shape[:2] else self.mask
        rho[mask] = self.rho_value
        u[mask, :] = 0.0
        u2 = u[:,:,0]**2 + u[:,:,1]**2
        feq = self.collisionOperator.compute_feq(rho, u, mask)
        f[mask] = feq[mask]


# Advection-diffusion operators

class PulsedConcentrationDirichletOperatorJax(PulsedConcentrationDirichletOperator):
    def __init__(self, descriptor, mask, base_value, pulse_value, t_start=0, t_end=None, sharpness=10.0):
        self.e = descriptor.e
        self.w = descriptor.w
        self.Q = descriptor.Q
        self.mask = mask
        self.base_value = base_value
        self.pulse_value = pulse_value
        self.t = 0
        self.t_start = t_start
        self.t_end = t_end
        self.sharpness = sharpness

    def __call__(self, g, u, phi):
        mask = np.transpose(self.mask) if self.mask.shape != phi.shape else self.mask

        # Smooth pulse using tanh
        if self.t_start <= self.t <= self.t_end:
            value = self.pulse_value
        else:
            value = self.base_value

        phi[mask] = value

        for i in range(self.Q):
            cu = u[:, :, 0]*self.e[i, 0] + u[:, :, 1]*self.e[i, 1]
            geq = self.w[i] * phi * (1 + 3*cu)
            g[mask, i] = geq[mask]
        #print(str(self.t) +" ww "+str(self.t_end)+ " ww " +str(value))
        self.t += 1


class ConstantScalarDirichletOperatorJax(ConstantScalarDirichletOperator):
    def __init__(self, descriptor, mask, value):
        self.e = descriptor.e
        self.w = descriptor.w
        self.Q = descriptor.Q
        self.mask = mask
        self.value = value

    def __call__(self, g, u, phi):
        mask = self.mask
        if mask.shape != phi.shape:
            mask = mask.T

        # Set scalar field
        phi[mask] = self.value

        # Compute equilibrium and update distribution
        for i in range(self.Q):
            cu = u[:, :, 0] * self.e[i, 0] + u[:, :, 1] * self.e[i, 1]
            geq = self.w[i] * phi * (1 + 3 * cu)
            g[mask, i] = geq[mask]


class ZeroGradientOutletOperatorJax(ZeroGradientOutletOperator):
    def __init__(self, descriptor, mask):
        self.e = descriptor.e
        self.w = descriptor.w
        self.Q = descriptor.Q
        self.mask = mask

    def __call__(self, g, u, phi):
        # Neumann boundary -> Boundary cell has to be the same value as the neighboring cell (is this cheating? Idk but it seems to work)
        mask = np.transpose(self.mask) if self.mask.shape != phi.shape else self.mask

        # Buffer our scalar value to apply our neighbors phi
        shifted_phi = np.roll(phi, shift=-1, axis=1)
        phi[mask] = shifted_phi[mask]

        # Compute equilibrium and update distribution
        for i in range(self.Q):
            cu = u[:, :, 0]*self.e[i, 0] + u[:, :, 1]*self.e[i, 1]
            geq = self.w[i] * phi * (1 + 3*cu)
            g[mask, i] = geq[mask]

# Collision operators

class BGK_collisionOperatorJax(BGK_collisionOperator[NDArray]):
    def __init__(self, tau: float, descriptor: LatticeDescriptor[NDArray]):
        self.tau = tau
        self.descriptor = descriptor
    
    def compute_feq(self, rho: NDArray, u: NDArray, mask: NDArray | None = None) -> NDArray:
        # more vectorized implementation
        u2  = np.sum(u**2, axis=-1)
        feq = np.matmul(u, self.descriptor.e.T)
        feq = self.descriptor.w[None, None, :] * rho[:, :, None] * (1 + 3*feq + 4.5*feq**2 - 1.5*u2[:, :, None])

        # apply mask, if provided
        return np.where(mask[:, :, None], feq, 0.0) if mask is not None else feq
    
    def compute_delta_f(self, f_lattice, rho, u, mask=None) -> NDArray:
        """ Relaxation Approach, this implementation kinda sucks atm because for 
        other collision operators (TRT, MRT) the Lattice2D Structure has to be changed
        -- but it works atm so idgaf """
        feq = self.compute_feq(rho, u, mask)
        delta_f = -(1.0 / self.tau) * (f_lattice - feq)
        return delta_f

class BGK_AdvectionDiffusion_collisionOperatorJax(BGK_AdvectionDiffusion_collisionOperator[NDArray]):
    def __init__(self, tau: float, descriptor: LatticeDescriptor[NDArray]):
        self.tau = tau

        self.descriptor = descriptor
        self.Q = descriptor.Q
        self.e = descriptor.e
        self.w = descriptor.w
        self.opp = descriptor.opp

    def compute_feq(self, phi, u, mask = None) -> NDArray:
        # more vectorized implementation
        feq = np.matmul(u, self.descriptor.e.T)
        feq = self.descriptor.w[None, None, :] * phi[:, :, None] * (1 + 3*feq)
        
        # apply mask, if provided
        return np.where(mask[:, :, None], feq, 0.0) if mask is not None else feq
    
    def compute_delta_f(self, f_lattice, phi, u, mask = None) -> NDArray:
        """ Relaxation Approach, this implementation kinda sucks atm because for 
        other collision operators (TRT, MRT) the Lattice2D Structure has to be changed
        -- but it works atm so idgaf """
        feq = self.compute_feq(phi, u, mask)
        delta_f = -(1.0 / self.tau) * (f_lattice - feq)
        return delta_f