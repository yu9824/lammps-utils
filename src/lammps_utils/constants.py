"""
Constants and type definitions for working with atomic data and LAMMPS data file formatting.

This module provides:
- The number of known chemical elements,
- A mapping of element symbols to their atomic masses,
- Type definitions for parsing atom and bond sections in LAMMPS data files.
"""

from rdkit import Chem

__all__ = (
    "N_ELEMENTS",
    "MAP_ELEMENT_MASSES",
    "COLS_ATOMS_LAMMPS_DATA_DTYPE",
    "COLS_BONDS_LAMMPS_DATA_DTYPE",
)

N_ELEMENTS = 118
"""Total number of elements in the periodic table. The 118th element is Oganesson (Og)."""


MAP_ELEMENT_MASSES: dict[str, float] = {
    _atom.GetSymbol(): _atom.GetMass()
    for _atom in (Chem.Atom(_i) for _i in range(1, N_ELEMENTS + 1))
}
"""Mapping from element symbols (e.g., 'H', 'C', 'O') to their atomic masses in atomic mass units (amu or g/mol)."""

COLS_ATOMS_LAMMPS_DATA_DTYPE = {
    "id": int,
    "mol": int,
    "type": int,
    "charge": float,
    "x": float,
    "y": float,
    "z": float,
}
"""Column name to data type mapping for atoms in a LAMMPS data file."""


COLS_BONDS_LAMMPS_DATA_DTYPE = {
    "id": int,
    "type": int,
    "atom1": int,
    "atom2": int,
}
"""Column name to data type mapping for bonds in a LAMMPS data file."""
