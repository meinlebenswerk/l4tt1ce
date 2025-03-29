# Numpy implementations of operators

from jax import Array
import jax.numpy as jnp

from dataclasses import dataclass

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

    def __call__(self, f: Array, u: Array, rho: Array):
        mask = jnp.transpose(self.mask) if self.mask.shape != f.shape[:2] else self.mask
        for i in range(len(self.opp)):
            f = f.at[mask, i].set(f[mask, self.opp[i]])
        return f, u, rho


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
        mask = jnp.transpose(self.mask) if self.mask.shape != f.shape[:2] else self.mask
        u = u.at[mask].set(self.velocity_func(u[mask].shape))
        rho = rho.at[mask].set(1.0) # assume constant pressure
        feq = self.collisionOperator.compute_feq(rho, u, mask)
        f = f.at[mask].set(feq[mask])
        return f, u, rho


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
        mask = jnp.transpose(self.mask) if self.mask.shape != f.shape[:2] else self.mask
        rho = rho.at[mask].set(self.rho_value)
        u = u.at[mask].set(0.0)
        feq = self.collisionOperator.compute_feq(rho, u, mask)
        f = f.at[mask].set(feq[mask])
        return f, u, rho


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

    def __call__(self, g: Array, u: Array, phi: Array):
        mask = jnp.transpose(self.mask) if self.mask.shape != phi.shape else self.mask

        # Smooth pulse using tanh
        if self.t_start <= self.t <= self.t_end:
            value = self.pulse_value
        else:
            value = self.base_value
        
        phi = phi.at[mask].set(value)

        cu = jnp.matmul(u, self.e.T) # (ny, nx, Q)
        geq = self.w[None, None, :] * phi[:, :, None] * (1 + 3*cu)
        g = g.at[mask, :].set(geq[mask])

        # for i in range(self.Q):
        #     cu = u[:, :, 0]*self.e[i, 0] + u[:, :, 1]*self.e[i, 1]
        #     geq = self.w[i] * phi * (1 + 3*cu)
        #     g[mask, i] = geq[mask]
        # #print(str(self.t) +" ww "+str(self.t_end)+ " ww " +str(value))
        self.t += 1

        return g, u, phi


class ConstantScalarDirichletOperatorJax(ConstantScalarDirichletOperator):
    def __init__(self, descriptor, mask, value):
        self.e = descriptor.e
        self.w = descriptor.w
        self.Q = descriptor.Q
        self.mask = mask
        self.value = value

    def __call__(self, g: Array, u: Array, phi: Array):
        mask = self.mask
        if mask.shape != phi.shape:
            mask = mask.T

        # Set scalar field
        phi = phi.at[mask].set(self.value)
        # phi[mask] = self.value

        # Compute equilibrium and update distribution
        cu = jnp.matmul(u, self.e.T) # (ny, nx, Q)
        geq = self.w[None, None, :] * phi[:, :, None] * (1 + 3*cu)
        g = g.at[mask, :].set(geq[mask])

        return g, u, phi

class ZeroGradientOutletOperatorJax(ZeroGradientOutletOperator):
    def __init__(self, descriptor, mask):
        self.e = descriptor.e
        self.w = descriptor.w
        self.Q = descriptor.Q
        self.mask = mask

    def __call__(self, g: Array, u: Array, phi: Array):
        # Neumann boundary -> Boundary cell has to be the same value as the neighboring cell (is this cheating? Idk but it seems to work)
        mask = jnp.transpose(self.mask) if self.mask.shape != phi.shape else self.mask

        # Buffer our scalar value to apply our neighbors phi
        shifted_phi = jnp.roll(phi, shift=-1, axis=1)
        # phi[mask] = shifted_phi[mask]
        phi = phi.at[mask].set(shifted_phi[mask])

        # Compute equilibrium and update distribution
        # isn't there an operator already for this somewhere?
        
        cu = jnp.matmul(u, self.e.T) # (ny, nx, Q)
        geq = self.w[None, None, :] * phi[:, :, None] * (1 + 3*cu)
        g = g.at[mask, :].set(geq[mask])
        # for i in range(self.Q):
        #     cu = u[:, :, 0]*self.e[i, 0] + u[:, :, 1]*self.e[i, 1]
        #     geq = self.w[i] * phi * (1 + 3*cu)
        #     g[mask, i] = geq[mask]
        return g, u, phi

# Collision operators

class BGK_collisionOperatorJax(BGK_collisionOperator[Array]):
    def __init__(self, tau: float, descriptor: LatticeDescriptor[Array]):
        self.tau = tau
        self.descriptor = descriptor
    
    def compute_feq(self, rho: Array, u: Array, mask: Array | None = None) -> Array:
        # more vectorized implementation
        u2  = jnp.sum(u**2, axis=-1)
        feq = jnp.matmul(u, self.descriptor.e.T)
        feq = self.descriptor.w[None, None, :] * rho[:, :, None] * (1 + 3*feq + 4.5*feq**2 - 1.5*u2[:, :, None])

        # apply mask, if provided
        return jnp.where(mask[:, :, None], feq, 0.0) if mask is not None else feq
    
    def compute_delta_f(self, f_lattice, rho, u, mask=None) -> Array:
        """ Relaxation Approach, this implementation kinda sucks atm because for 
        other collision operators (TRT, MRT) the Lattice2D Structure has to be changed
        -- but it works atm so idgaf """
        feq = self.compute_feq(rho, u, mask)
        delta_f = -(1.0 / self.tau) * (f_lattice - feq)
        return delta_f

class BGK_AdvectionDiffusion_collisionOperatorJax(BGK_AdvectionDiffusion_collisionOperator[Array]):
    def __init__(self, tau: float, descriptor: LatticeDescriptor[Array]):
        self.tau = tau

        self.descriptor = descriptor
        self.Q = descriptor.Q
        self.e = descriptor.e
        self.w = descriptor.w
        self.opp = descriptor.opp

    def compute_feq(self, phi, u, mask = None) -> Array:
        # more vectorized implementation
        feq = jnp.matmul(u, self.descriptor.e.T)
        feq = self.descriptor.w[None, None, :] * phi[:, :, None] * (1 + 3*feq)
        
        # apply mask, if provided
        return jnp.where(mask[:, :, None], feq, 0.0) if mask is not None else feq
    
    def compute_delta_f(self, f_lattice, phi, u, mask = None) -> Array:
        """ Relaxation Approach, this implementation kinda sucks atm because for 
        other collision operators (TRT, MRT) the Lattice2D Structure has to be changed
        -- but it works atm so idgaf """
        feq = self.compute_feq(phi, u, mask)
        delta_f = -(1.0 / self.tau) * (f_lattice - feq)
        return delta_f