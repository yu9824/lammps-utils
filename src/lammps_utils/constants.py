"""
Constants for working with atomic data and LAMMPS data file formatting.

This module provides:
- The number of known chemical elements,
- A mapping of element symbols to their atomic masses,
- Data type mappings for parsing atom and bond sections in LAMMPS data files.
"""

from rdkit import Chem

__all__ = (
    "AVOGADRO",
    "N_ELEMENTS",
    "MAP_ELEMENT_MASSES",
    "COLS_ATOMS_LAMMPS_DATA_DTYPE",
    "COLS_BONDS_LAMMPS_DATA_DTYPE",
    "MAP_VDW_RADIUS",
    "COLS_XYZ",
)

AVOGADRO = 6.022e23
"""Avogadro's number (6.022 × 10²³) in units of mol⁻¹.

This constant represents the number of atoms, molecules, or other particles
in one mole of a substance. Used for converting between atomic mass units
and grams, and for density calculations.
"""

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

MAP_VDW_RADIUS = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}
"""Mapping from atomic symbols to van der Waals radii in angstroms."""

COLS_XYZ = ["x", "y", "z"]
"""Coordinate column labels

``["x", "y", "z"]``
"""
