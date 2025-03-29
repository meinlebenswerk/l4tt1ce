from src.lbm_engine.impl.numpy.descriptor import D2Q9Numpy
from src.lbm_engine.impl.numpy.lattice import Lattice2DNumpy, ScalarLattice2DNumpy
from src.lbm_engine.impl.numpy.operator import (
    BounceBackOperatorNumpy,
    VelocityDirichletOperatorNumpy,
    PressureDirichletOperatorNumpy,
    PulsedConcentrationDirichletOperatorNumpy,
    ConstantScalarDirichletOperatorNumpy,
    ZeroGradientOutletOperatorNumpy,
    BGK_collisionOperatorNumpy,
    BGK_AdvectionDiffusion_collisionOperatorNumpy,
)

# re-export the classes for easier, namespaced backend access
D2Q9 = D2Q9Numpy

Lattice2D = Lattice2DNumpy
ScalarLattice2D = ScalarLattice2DNumpy

BounceBackOperator = BounceBackOperatorNumpy
VelocityDirichletOperator = VelocityDirichletOperatorNumpy
PressureDirichletOperator = PressureDirichletOperatorNumpy

PulsedConcentrationDirichletOperator = PulsedConcentrationDirichletOperatorNumpy
ConstantScalarDirichletOperator = ConstantScalarDirichletOperatorNumpy
ZeroGradientOutletOperator = ZeroGradientOutletOperatorNumpy

BGK_collisionOperator = BGK_collisionOperatorNumpy
BGK_AdvectionDiffusion_collisionOperator = BGK_AdvectionDiffusion_collisionOperatorNumpy

__all__ = [
    "Lattice2D",
    "ScalarLattice2D",
    "BounceBackOperator",
    "VelocityDirichletOperator",
    "PressureDirichletOperator",
    "PulsedConcentrationDirichletOperator",
    "ConstantScalarDirichletOperator",
    "ZeroGradientOutletOperator",
    "BGK_collisionOperator",
    "BGK_AdvectionDiffusion_collisionOperator"
]