import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem


def _apply_minimum_image_convention(
    conformer: Chem.rdchem.Conformer, cell_size: ArrayLike
) -> None:
    """
    Applies the minimum image convention to a conformer's atom positions
    to ensure continuity across periodic boundary conditions.

    Parameters
    ----------
    conformer : Chem.rdchem.Conformer
        RDKit conformer object containing 3D coordinates.
    cell_size : ArrayLike
        Size of the periodic box as a 1D array-like object of shape (3,).
    """
    cell_size = np.asarray(cell_size)
    assert cell_size.ndim == 1
    assert cell_size.shape[0] == 3

    pos: np.ndarray = conformer.GetPositions()  # shape: (N_atoms, 3)
    new_pos = np.copy(pos)

    for idx_atom in range(1, pos.shape[0]):
        ref: np.ndarray = new_pos[idx_atom - 1]
        vec: np.ndarray = pos[idx_atom] - ref
        corrected_vec = vec - cell_size * np.round(vec / cell_size)
        new_pos[idx_atom] = ref + corrected_vec

        conformer.SetAtomPosition(
            idx_atom, new_pos[idx_atom]
        )  # update coordinate


def fix_molecule_periodic_boundary(
    mol: Chem.rdchem.Mol, cell_size: ArrayLike, confId: int = 0
) -> Chem.rdchem.Mol:
    """
    Adjusts atom positions of a molecule's conformer to respect periodic boundary
    conditions using the minimum image convention.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        RDKit Mol object containing at least one conformer.
    cell_size : ArrayLike
        Periodic box dimensions as a 1D array-like object of shape (3,).
    confId : int, optional
        ID of the conformer to adjust. Default is 0.

    Returns
    -------
    Chem.rdchem.Mol
        A copy of the input molecule with adjusted atom positions.
    """
    mol = Chem.Mol(mol)  # make a copy to avoid modifying the original
    conf = mol.GetConformer(confId)

    _apply_minimum_image_convention(conf, np.asarray(cell_size))
    return mol
