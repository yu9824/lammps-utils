import numpy as np
from rdkit import Chem


def calculate_rg(mol: Chem.rdchem.Mol, confId: int = -1, mode="mass") -> float:
    if mol.GetNumConformers() == 0:
        raise ValueError("No conformer")

    conf = mol.GetConformer(confId)
    positions = conf.GetPositions()

    if mode == "mass":
        weights = tuple(atom.GetMass() for atom in mol.GetAtoms())
    elif mode == "geometry":
        weights = None
    else:
        raise ValueError

    center = np.average(positions, axis=0, weights=weights, keepdims=True)
    return np.square(
        np.average(np.square(positions - center), axis=0, weights=weights)
    ).item()
