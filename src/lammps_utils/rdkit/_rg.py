from typing import Literal

import numpy as np
from rdkit import Chem


def compute_rg(
    mol: Chem.rdchem.Mol,
    confId: int = -1,
    mode: Literal["mass", "geometry"] = "mass",
    removeHs: bool = False,
) -> float:
    """
    Compute the radius of gyration (Rg) of a molecule.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The molecule for which Rg is computed.
    confId : int, optional
        Index of the conformer to use (default is -1, which selects the first conformer).
    mode : {'mass', 'geometry'}, optional
        Mode of computation:
        - 'mass': Compute Rg based on atomic masses (default).
        - 'geometry': Compute Rg based solely on atomic positions.
    removeHs : bool, optional
        Whether to remove hydrogens from the molecule before computation (default is False).

    Returns
    -------
    float
        The radius of gyration (Rg) of the molecule.

    Raises
    ------
    ValueError
        If no conformer is available for the molecule.
        If an invalid mode is specified.

    Notes
    -----
    - If mode='mass', Rg is computed using atomic masses as weights.
    - If mode='geometry', Rg is computed based only on atomic positions.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("No conformer")

    if removeHs:
        mol = Chem.RemoveHs(mol)

    conf = mol.GetConformer(confId)
    positions = conf.GetPositions()

    if mode == "mass":
        weights = tuple(
            atom.GetMass()
            for atom in mol.GetAtoms()
            if isinstance(atom, Chem.Atom)
        )
    elif mode == "geometry":
        weights = None
    else:
        raise ValueError

    center = np.average(positions, axis=0, weights=weights, keepdims=True)
    return np.sqrt(
        np.average(
            np.sum(np.square(positions - center), axis=1),
            axis=0,
            weights=weights,
        )
    ).item()
