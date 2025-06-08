"""
High-level RDKit-based utilities for molecular structure processing.

This submodule includes:
- Bond order estimation based on atomic distance and element types,
- Coordinate unwrapping for RDKit molecules under periodic boundary conditions (PBC),
- Main chain detection in polymer-like molecular structures.
"""

from ._bond import get_bond_order
from ._density import compute_density
from ._ffv import compute_ffv
from ._pbc import unwrap_rdkit_mol_under_pbc
from ._poly import find_main_chains
from ._rg import compute_rg

__all__ = (
    "get_bond_order",
    "compute_density",
    "compute_ffv",
    "unwrap_rdkit_mol_under_pbc",
    "find_main_chains",
    "compute_rg",
)
