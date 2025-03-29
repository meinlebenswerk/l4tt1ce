import time


implementation = "jax"  # Change this to "numpy" if you want to use the numpy implementation

if implementation == "numpy":
    import numpy as np
    from src.lbm_engine.impl.numpy import (
        D2Q9,
        Lattice2D, ScalarLattice2D,
        BounceBackOperator, PressureDirichletOperator, VelocityDirichletOperator, ZeroGradientOutletOperator, ConstantScalarDirichletOperator,
        BGK_collisionOperator,
        BGK_AdvectionDiffusion_collisionOperator
    )
    from src.lbm_engine.impl.numpy.geometry import create_triangle_mask, create_rectangle_mask
elif implementation == "jax":
    import jax
    import jax.numpy as np
    from src.lbm_engine.impl.jax import (
        D2Q9,
        Lattice2D, ScalarLattice2D,
        BounceBackOperator, PressureDirichletOperator, VelocityDirichletOperator, ZeroGradientOutletOperator, ConstantScalarDirichletOperator,
        BGK_collisionOperator,
        BGK_AdvectionDiffusion_collisionOperator
    )
    from src.lbm_engine.impl.jax.geometry import create_triangle_mask, create_rectangle_mask
else:
    raise ValueError("Invalid implementation selected. Choose either 'numpy' or 'jax'.")

from src.lbm_engine.utils import PrintLatticeInformation
from src.lbm_engine.core.visualization import export_fields_vti, visualize_combined

""" This is a basic demonstration of the implementation that showcases a heated object in a channel.
It is created using Navier Stokes Lattice one way coupled to an Advection Diffusion Lattice 
using the Velocity Field of the NS lattice. It creates combined plots of velocity and scalar Value 
as well as VTK files for visualization in Paraview """

# Basic Simulation Parameters
timesteps = 20000
plot_interval = 100

# Navier Stokes Setup
tau_NS = 0.65
u_max = 0.1

# Advection Diffusion Setup
tau_diff = 1.5 # this directly influences our diffusivity due to BGK AD Dynamics has to be >0.5 to be stable
wallTemperature = 1 # this is the scalar value for our wall temperature
objectTemperature = 3

# Geometry Setup
nx, ny = 300, 100
X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

# This looks pretty goofy but it creates a simple simulation geometry so the output does not look as boring
object_mask = (create_rectangle_mask(X, Y, center_x=nx/3, center_y=ny/4, width=10, height=50) 
| create_rectangle_mask(X, Y, center_x=(nx/3)+nx/8, center_y=ny-(ny/4), width=10, height=50)
| create_rectangle_mask(X, Y, center_x=nx/3+nx/3+nx/8, center_y=ny/4, width=10, height=50))

# Here we create a triangular mask for the heated object using a geometric helper method
hot_object_mask = create_triangle_mask(
    X, Y,
    center_x=nx // 8,
    center_y=ny // 2,
    base_width=ny // 5,
    height=nx // 10,
    direction='right'  
)


## Navier Stokes Lattice Setup
def setupNavierStokesLattice():
    """ this is a basic setup method, that returns our NS-Lattice. We apply bounceback to the walls and all the objects 
    and set some basic Boundary Conditions for inlet and outlet """
    print("Setting up Navier Stokes Lattice...")
    # Definition of Descriptor and Collision Operator
    descriptorNS = D2Q9()
    collisionOperatorBGK = BGK_collisionOperator(tau=tau_NS, descriptor=descriptorNS)

    # Create Lattice
    NSLattice = Lattice2D(
        nx=nx,
        ny=ny,
        descriptor=D2Q9(),
        collisionOperator=collisionOperatorBGK,
    )
    # We add bounceback operator to the top and bottom walls to build the pipe geometry
    # We use a combination of the masks in order to add bounceback on the objects as well
    top_wall_mask = (Y == ny - 1)
    bottom_wall_mask = (Y == 0)
    wall_mask = top_wall_mask | bottom_wall_mask | object_mask | hot_object_mask
    NSLattice.addOperator("obstacle", BounceBackOperator(NSLattice.descriptor, wall_mask))

    # Velocity inlet on the left (Dirichlet Velocity Boundary)
    inlet_mask = X == 0
    def inlet_velocity(shape):
        return np.zeros(shape) + np.array([u_max, 0])
    NSLattice.addOperator("inlet", VelocityDirichletOperator(NSLattice.descriptor, collisionOperatorBGK, inlet_mask, inlet_velocity))

    # Constant pressure Dirichlet on the outlet (right)
    outlet_mask = X == nx - 1
    NSLattice.addOperator("outlet", PressureDirichletOperator(NSLattice.descriptor, collisionOperatorBGK, outlet_mask, rho_value=1.0))

    return NSLattice

def setupAdvectionDiffusionLattice():
    """ This is the setup method for the AD-Lattice. The heated object is assigned a fixed scalar value (temperature) 
    and we create a cold air inlet on the left using a Dirichlet Constant Scalar Boundary. All the Walls are also fitted
    with constant Scalar Dirichlet because otherwise this simulation would be completely Periodic in all directions.
    I know this is ugly but the implementation using np.roll is inherently periodic so we have to cheat a little to apply
    fixed boundary conditions at our simulation edges. The outlet is modeled using a Neumann boundary."""
    print("Setting up Advection Diffusion Lattice...")
    # Definition of Descriptor and Collision Operator
    descriptorAD = D2Q9()
    collisionOperatorBGK_AD = BGK_AdvectionDiffusion_collisionOperator(tau=tau_diff, descriptor=descriptorAD)

    # Initialize AD Lattice
    ADLattice = ScalarLattice2D(
        nx=nx,
        ny=ny,
        descriptor=D2Q9(),
        collisionOperator=collisionOperatorBGK_AD
    )
    #we add a cold air inlet on the left wall
    mask = X == 0
    ADLattice.addOperator("coldAirInlet", ConstantScalarDirichletOperator(descriptorAD,mask,1.))
    # the outlet is located on the right wall as a Neumann Zero Gradient Boundary
    outlet_mask = X == nx - 1
    ADLattice.addOperator("outlet_phi", ZeroGradientOutletOperator(descriptorAD, outlet_mask))
    # we apply cold to our walls
    top_wall_mask = (Y == ny - 1)
    bottom_wall_mask = (Y == 0)
    wall_mask = top_wall_mask | bottom_wall_mask | object_mask
    ADLattice.addOperator("walls", ConstantScalarDirichletOperator(descriptorAD, wall_mask, wallTemperature))
    # we add a constant scalar dirichlet boundary to the hot object in the channel
    ADLattice.addOperator("hotObject", ConstantScalarDirichletOperator(descriptorAD, hot_object_mask, objectTemperature))
    return ADLattice

ADLattice = setupAdvectionDiffusionLattice()
NSLattice = setupNavierStokesLattice()

# setup a jit-compilable step-function
if implementation == "jax":
    def step_lattice(NSLattice, ADLattice, num: int = 1): # type: ignore
        for i in range(num):
            # Stream both lattices
            NSLattice.step()
            ADLattice.u = NSLattice.u
            ADLattice.step()
        return NSLattice, ADLattice
else:
    def step_lattice(NSLattice, ADLattice, num: int = 1): # type: ignore
        for i in range(1):
            # Stream both lattices
            NSLattice.step()
            ADLattice.u = NSLattice.u
            ADLattice.step()
        return NSLattice, ADLattice

t = 0
num_step = 100

# Main Simulation Loop
while t < timesteps:
    time_in = time.time()
    # we stream & collide both lattices and apply the velocity field from the NS Lattice to the AD Lattice
    NSLattice, ADLattice = step_lattice(NSLattice, ADLattice, num=num_step)
    time_out = time.time()
    t += num_step

    if t % plot_interval == 0:
            avg_time = (time_out - time_in) / num_step
            print(f"Average time per step: {avg_time:.4f} seconds | {1.0/avg_time:.2f} steps/s")
            print("Timestep " + str(t) + " / " + str(timesteps))
            # Save the VTI Files for Paraview
            export_fields_vti(t, ADLattice.u, ADLattice.phi)
            # Save the Combined plot for velocity and Phi
            visualize_combined(ADLattice, t, overall_title="Heated Object in Channel")
            # cool little method for plotting our Lattice Information (more of a gimmic)
            PrintLatticeInformation(t, NSLattice, ADLattice)

