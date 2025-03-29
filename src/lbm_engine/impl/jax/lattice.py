# Numpy implementations of the lattice classes
import jax
import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

from dataclasses import dataclass

from src.lbm_engine.core.descriptor import LatticeDescriptor
from src.lbm_engine.core.lattice import Lattice2D, ScalarLattice2D
from src.lbm_engine.core.operators.collision import CollisionOperator, ADCollisionOperator
from src.lbm_engine.core.operators.navier_stokes import NSOperator
from src.lbm_engine.core.operators.advection_diffusion import ADOperator


""" Dataclasses for jax' stateless programming model """

@dataclass
class Lattice2DState:
    rho: jnp.ndarray
    u: jnp.ndarray
    f: jnp.ndarray
    t: int

@dataclass
class Lattice2DConfig:
    nx: int
    ny: int
    e: jnp.ndarray  # shape (Q, 2)
    opp: jnp.ndarray
    Q: int
    operators: dict[str, NSOperator]
    collision_operator: CollisionOperator

@dataclass
class ScalarLattice2DState:
    u: jnp.ndarray
    phi: jnp.ndarray
    g: jnp.ndarray
    t: int

@dataclass
class ScalarLattice2DConfig:
    nx: int
    ny: int
    e: jnp.ndarray  # shape (Q, 2)
    opp: jnp.ndarray
    Q: int
    operators: dict[str, ADOperator]
    collision_operator: ADCollisionOperator

""" Jax 2D Lattice step function """

# @jax.jit
def lattice_2d_step(state: Lattice2DState, config: Lattice2DConfig):

    # ## Streaming Step
    # # for all Q directions we stream the distribution function across the lattice using our discrete velocity set ei
    # # looks complicated af but is actually not that hard
    # # this is periodic -> if you dont want this you have to continously overwrite the boundary conditions
    # for i in range(self.Q):
    #     self.f[:, :, i] = jnp.roll(
    #         jnp.roll(self.f[:, :, i], self.e[i, 0], axis=1),
    #         self.e[i, 1], axis=0
    #     )

    def stream_fn(i, f):
        f_i = jnp.roll(f[:, :, i], config.e[i, 0], axis=1)
        f_i = jnp.roll(f_i, config.e[i, 1], axis=0)
        return f.at[:, :, i].set(f_i)
    
    f = jax.lax.fori_loop(0, config.Q, stream_fn, state.f)
    u = state.u
    rho = state.rho

    # apply boundary conditions
    for operator in config.operators.values():
        f, u, rho = operator(f, u, rho)

    # compute updates
    rho = jnp.sum(f, axis=2)
    u_x = jnp.sum(f * config.e[:, 0], axis=2) / rho
    u_y = jnp.sum(f * config.e[:, 1], axis=2) / rho
    u = jnp.stack((u_x, u_y), axis=2)

    # calculate delta f from collision Operator
    delta_f = config.collision_operator.compute_delta_f(f, rho, u)
    f = f + delta_f

    return Lattice2DState(
        rho=rho,
        u=u,
        f=f,
        t=state.t + 1
    )

# @jax.jit
def scalar_lattice_2d_step(state: ScalarLattice2DState, config: ScalarLattice2DConfig):
    def stream_fn(i, g):
        g_i = jnp.roll(g[:, :, i], config.e[i, 0], axis=1)
        g_i = jnp.roll(g_i, config.e[i, 1], axis=0)
        return g.at[:, :, i].set(g_i)

    g = jax.lax.fori_loop(0, config.Q, stream_fn, state.g)
    phi = jnp.sum(g, axis=2)

    for op in config.operators.values():
        g, state.u, phi = op(g, state.u, phi)  # external & pure

    delta_g = config.collision_operator.compute_delta_f(g, phi, state.u)
    g = g + delta_g

    return ScalarLattice2DState(phi=phi, u=state.u, g=g, t=state.t + 1)

class Lattice2DJax(Lattice2D):
    """
    Jax implementation of a 2D lattice
    """

    def __init__(self, nx, ny, descriptor: LatticeDescriptor[jax.Array], collisionOperator: ADCollisionOperator):
        super().__init__()
        
        # save configuration
        self.nx, self.ny = nx, ny
        self.descriptor = descriptor
        self.collisionOperator = collisionOperator

        # defer init to first invocation of step-function, so all operators are captured
        # TODO: could override operators later
        self.config: Lattice2DConfig | None = None

        # setup initial state
        print("Type of collisionOperator:", type(self.collisionOperator))
        rho = jnp.ones((ny, nx))
        u = jnp.zeros((ny, nx, 2))
        f = collisionOperator.compute_feq(rho, u)
        self.state = Lattice2DState(
            rho=jnp.zeros((ny, nx)),
            u=u,
            f=f,
            t=0
        )

    @property
    def u_np(self) -> NDArray:
        return np.array(self.state.u, copy=False)
    
    @property
    def rho_np(self) -> NDArray:
        return np.array(self.state.rho, copy=False)
    
    def step(self):
        # initialize config on first invocation
        if self.config is None:
            self.config = Lattice2DConfig(
                nx=self.nx,
                ny=self.ny,
                e=self.descriptor.e,
                opp=self.descriptor.opp,
                Q=self.descriptor.Q,
                operators=self.operators,
                collision_operator=self.collisionOperator
            )

        # update state
        self.state = lattice_2d_step(self.state, self.config)


class ScalarLattice2DJax(ScalarLattice2D):
    """
    Jax implementation of a 2D scalar lattice
    """

    def __init__(self, nx, ny, descriptor, collisionOperator: ADCollisionOperator):
        super().__init__()

        # save configuration
        self.nx, self.ny = nx, ny
        self.descriptor = descriptor
        self.collisionOperator = collisionOperator

        # defer init to first invocation of step-function, so all operators are captured
        self.config: ScalarLattice2DConfig | None = None
        
        # setup initial state
        print("Type of collisionOperator:", type(self.collisionOperator))
        self.u = jnp.zeros((ny, nx, 2))
        self.phi = jnp.ones((ny, nx))
        self.g = collisionOperator.compute_feq(self.phi, self.u)
        self.state = ScalarLattice2DState(
            u=self.u,
            phi=self.phi,
            g=self.g,
            t=0
        )

    @property
    def phi_np(self) -> NDArray:
        return np.array(self.state.phi, copy=False)

    def step(self, n: int = 1):
        # initialize config on first invocation
        if self.config is None:
            self.config = ScalarLattice2DConfig(
                nx=self.nx,
                ny=self.ny,
                e=self.descriptor.e,
                opp=self.descriptor.opp,
                Q=self.descriptor.Q,
                operators=self.operators,
                collision_operator=self.collisionOperator
            )

        # update state
        self.state = scalar_lattice_2d_step(self.state, self.config)
