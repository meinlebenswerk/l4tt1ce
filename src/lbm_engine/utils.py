# Utility functions
from src.lbm_engine.core.lattice import Lattice2D, ScalarLattice2D

def PrintLatticeInformation(t: int, nslattice: Lattice2D, adlattice: ScalarLattice2D | None = None) -> None:
    print("\n" + "=" * 60)
    print("LATTICE INFORMATION".center(60))
    print(f"Time step: t = {t}".center(60))
    print("-" * 60)

    # --- Navier-Stokes Lattice ---
    nslattice.print()
    
    # --- Advection-Diffusion Lattice ---

    if adlattice is not None:
        adlattice.print()

    print("=" * 60 + "\n")
