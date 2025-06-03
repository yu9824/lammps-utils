from typing import Literal

import numpy as np
from rdkit import Chem


def compute_rg(
    mol: Chem.rdchem.Mol,
    confId: int = -1,
    mode: Literal["mass", "geometry"] = "mass",
    removeHs: bool = False,
) -> float:
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
