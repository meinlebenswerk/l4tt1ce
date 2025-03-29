# Numpy implementations of the lattice classes
import jax
import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp


from src.lbm_engine.core.lattice import Lattice2D, ScalarLattice2D
from src.lbm_engine.core.operators.collision import CollisionOperator


class Lattice2DJax(Lattice2D):
    """
    Numpy implementation of a 2D lattice
    """

    def __init__(self, nx, ny, descriptor, collisionOperator):
        super().__init__()
        #define the lattice dimensions in lattice units
        self.nx, self.ny = nx, ny 
        self.descriptor = descriptor
        #set the operator used for all colission on this lattice
        self.collisionOperator = collisionOperator
        #get the velocity directions from our lattice descriptor
        self.e = descriptor.e
        #get the direction opposites from descriptor definition
        self.opp = descriptor.opp
        #get the unique velocity count from descriptor definition
        self.Q = descriptor.Q

        ##Setup of all the fields needed
        self.X, self.Y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        #setup density at 1
        self.rho = jnp.ones((ny, nx))
        #setup velocity as 0 in all directions (2 for xy)
        self.u = jnp.zeros((ny, nx, 2))
        #setup distribution function as 0 everywhere for all Q directions
        self.f = jnp.zeros((ny, nx, self.Q))
        #setup of timeStep variable
        self.t = 0

        ##setup the distribution function as equilibrium using the collision operator
        print("Type of collisionOperator:", type(self.collisionOperator))
        self.f = self.collisionOperator.compute_feq(self.rho, self.u)

    @property
    def u_np(self) -> NDArray:
        return np.array(self.u, copy=False)
    
    @property
    def rho_np(self) -> NDArray:
        return np.array(self.rho, copy=False)
    
    @jax.jit
    def step(self):
        ## Streaming Step
        # for all Q directions we stream the distribution function across the lattice using our discrete velocity set ei
        # looks complicated af but is actually not that hard
        # this is periodic -> if you dont want this you have to continously overwrite the boundary conditions
        for i in range(self.Q):
            self.f[:, :, i] = jnp.roll(
                jnp.roll(self.f[:, :, i], self.e[i, 0], axis=1),
                self.e[i, 1], axis=0
            )
        
        ## Here we iterate over the operators in the operator List and call the apply method for each one to apply boundary conditions
        for operator in self.operators.values():
            operator(self.f, self.u, self.rho)
        
        # calculate density from distribution function 
        self.rho = jnp.sum(self.f, axis=2)
        # calculate x component of velocity set from the distribution function (use direction vectors as weights (TM1 flashback))
        self.u[:, :, 0] = jnp.sum(self.f * self.e[:, 0], axis=2) / self.rho
        # same for y component
        self.u[:, :, 1] = jnp.sum(self.f * self.e[:, 1], axis=2) / self.rho


        ## Collision Step
        # Calculate delta f from collision Operator
        self.f += self.collisionOperator.compute_delta_f(self.f, self.rho, self.u)


class ScalarLattice2DNumpy(ScalarLattice2D):
    """
    Numpy implementation of a 2D scalar lattice
    """

    def __init__(self, nx, ny, descriptor, collisionOperator: CollisionOperator):
        super().__init__()

        #define the lattice dimensions in lattice units
        self.nx, self.ny = nx, ny 
        self.descriptor = descriptor
        #set the operator used for all colission on this lattice
        self.collisionOperator = collisionOperator
        #get the velocity directions from our lattice descriptor
        self.e = descriptor.e
        #get the direction opposites from descriptor definition
        self.opp = descriptor.opp
        #get the unique velocity count from descriptor definition
        self.Q = descriptor.Q

        ##Setup of all the fields needed
        self.X, self.Y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        #setup velocity as 0 in all directions (2 for xy)
        self.u = jnp.zeros((ny, nx, 2))
        #setup scalar field as 1 everywhere
        self.phi = jnp.ones((ny, nx))
        #setup distribution function as 0 everywhere for all Q directions
        self.g= jnp.zeros((ny, nx, self.Q))
        #setup of timeStep variable
        self.t = 0

        ##setup the distribution function as equilibrium using the collision operator
        print("Type of collisionOperator:", type(self.collisionOperator))
        self.g = self.collisionOperator.compute_feq(self.phi, self.u)

    @property
    def phi_np(self) -> NDArray:
        return np.array(self.phi, copy=False)

    @jax.jit
    def step(self, n: int = 1):
        for i in range(self.Q):
            self.g[:, :, i] = jnp.roll(jnp.roll(self.g[:, :, i], self.e[i, 0], axis=1), self.e[i, 1], axis=0)

        self.phi = jnp.sum(self.g, axis=2)

        for operator in self.operators.values():
            operator(self.g, self.u, self.phi)

        self.g += self.collisionOperator.compute_delta_f(self.g, self.phi, self.u)
