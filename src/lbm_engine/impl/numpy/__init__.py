from src.lbm_engine.impl.numpy.descriptor import D2Q9Numpy
from src.lbm_engine.impl.numpy.lattice import Lattice2DJax, ScalarLattice2DJax
from src.lbm_engine.impl.numpy.operator import (
    BounceBackOperatorJax,
    VelocityDirichletOperatorJax,
    PressureDirichletOperatorJax,
    PulsedConcentrationDirichletOperatorJax,
    ConstantScalarDirichletOperatorJax,
    ZeroGradientOutletOperatorJax,
    BGK_collisionOperatorJax,
    BGK_AdvectionDiffusion_collisionOperatorJax,
)

# re-export the classes for easier, namespaced backend access
D2Q9 = D2Q9Numpy

Lattice2D = Lattice2DJax
ScalarLattice2D = ScalarLattice2DJax

BounceBackOperator = BounceBackOperatorJax
VelocityDirichletOperator = VelocityDirichletOperatorJax
PressureDirichletOperator = PressureDirichletOperatorJax

PulsedConcentrationDirichletOperator = PulsedConcentrationDirichletOperatorJax
ConstantScalarDirichletOperator = ConstantScalarDirichletOperatorJax
ZeroGradientOutletOperator = ZeroGradientOutletOperatorJax

BGK_collisionOperator = BGK_collisionOperatorJax
BGK_AdvectionDiffusion_collisionOperator = BGK_AdvectionDiffusion_collisionOperatorJax

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