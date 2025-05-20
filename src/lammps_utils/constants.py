from rdkit import Chem

__all__ = (
    "N_ELEMENTS",
    "MAP_ELEMENT_MASSES",
    "COLS_ATOMS_LAMMPS_DATA_DTYPE",
    "COLS_BONDS_LAMMPS_DATA_DTYPE",
)

N_ELEMENTS = 118
"""The number of elements in the periodic table. The last element is Oganesson (Og)."""


MAP_ELEMENT_MASSES: dict[str, float] = {
    _atom.GetSymbol(): _atom.GetMass()
    for _atom in (Chem.Atom(_i) for _i in range(1, N_ELEMENTS + 1))
}
"""A dictionary mapping element symbols to their atomic masses in amu (g/mol)."""

COLS_ATOMS_LAMMPS_DATA_DTYPE = {
    "id": int,
    "mol": int,
    "type": int,
    "charge": float,
    "x": float,
    "y": float,
    "z": float,
}
"""A dictionary mapping column names to their data types in the LAMMPS data file."""

COLS_BONDS_LAMMPS_DATA_DTYPE = {
    "id": int,
    "type": int,
    "atom1": int,
    "atom2": int,
}
