from src.lbm_engine.impl.jax.descriptor import D2Q9Jax
from src.lbm_engine.impl.jax.lattice import Lattice2DJax, ScalarLattice2DJax
from src.lbm_engine.impl.jax.operator import (
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
D2Q9 = D2Q9Jax

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