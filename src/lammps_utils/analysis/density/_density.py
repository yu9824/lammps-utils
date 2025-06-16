import numpy as np
import scipy.constants
from numpy.typing import ArrayLike
from rdkit import Chem


def compute_density(mol: Chem.rdchem.Mol, cell_size: ArrayLike) -> float:
    """
    Compute the mass density of a molecule in g/cm³ based on its atomic mass and a given cell size.

    This function assumes that the input cell size is given in angstroms (Å), as commonly used in
    molecular simulations such as LAMMPS with `units real` or `units metal`.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        RDKit molecule object. The atomic masses are obtained using `atom.GetMass()`, which returns values in amu.
    cell_size : ArrayLike
        A 1D array-like object of shape (3,) specifying the dimensions of the simulation cell (in angstroms).

    Returns
    -------
    float
        The computed density in grams per cubic centimeter (g/cm³).

    Notes
    -----
    - The atomic mass is converted from amu to grams using Avogadro's number.
    - The cell volume is converted from Å³ to cm³ using the relation: 1 Å = 1e-8 cm.
    """
    cell_size = np.asarray(cell_size)
    assert cell_size.ndim == 1
    assert cell_size.shape[0] == 3

    mass = (
        sum(
            atom.GetMass()
            for atom in mol.GetAtoms()
            if isinstance(atom, Chem.Atom)
        )
        / scipy.constants.Avogadro
    )
    volume = np.prod(
        cell_size * (scipy.constants.angstrom / scipy.constants.centi)
    )
    return mass / volume
