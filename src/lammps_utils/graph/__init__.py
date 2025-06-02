"""
Graph utilities for molecular structures.

This submodule provides functions for:
- Finding the farthest-apart atoms in a molecular graph,
- Identifying atoms that are part of ring structures,
- Unwrapping molecular coordinates under periodic boundary conditions (PBC).
"""

from ._main_chain import farthest_node_pair, nodes_in_cycles
from ._pbc import unwrap_molecule_under_pbc

__all__ = (
    "farthest_node_pair",
    "nodes_in_cycles",
    "unwrap_molecule_under_pbc",
)
