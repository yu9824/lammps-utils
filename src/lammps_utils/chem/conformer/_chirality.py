"""Utilities for inverting molecular chirality."""

from __future__ import annotations

from typing import Final

import numpy as np
from rdkit import Chem

from lammps_utils.io.mol import set_positions


_DEFAULT_CONF_ID: Final[int] = -1


def invert_chirality_coords(coords: np.ndarray) -> np.ndarray:
    """
    Invert chirality of coordinates while preserving the centroid.

    This function reflects a set of 3D coordinates through their geometric
    center. As a result, the overall handedness (chirality) of the structure
    is inverted while the centroid remains unchanged.

    Parameters
    ----------
    coords : (N, 3) np.ndarray
        Cartesian coordinates of atoms. Each row corresponds to one atom.

    Returns
    -------
    (N, 3) np.ndarray
        Reflected coordinates with inverted handedness relative to the
        geometric center.
    """
    coords = np.asarray(coords, dtype=float)
    center = coords.mean(axis=0)
    return -(coords - center) + center


def invert_chirality(
    mol: Chem.rdchem.Mol,
    conf_id: int = _DEFAULT_CONF_ID,
) -> Chem.rdchem.Mol:
    """
    Create a copy of a molecule with inverted chirality for one conformer.

    The function:

    1. Copies the input RDKit molecule.
    2. Extracts the coordinates of the specified conformer.
    3. Reflects the coordinates through their centroid using
       :func:`invert_chirality_coords`.
    4. Replaces the original conformer in the copied molecule with the
       reflected coordinates.
    5. Reassigns chiral tags from the 3D structure.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        Input RDKit molecule containing at least one conformer.
    conf_id : int, optional
        Conformer ID to invert. The default value ``-1`` selects the
        default RDKit conformer (typically the first conformer).

    Returns
    -------
    Chem.rdchem.Mol
        A new molecule with the specified conformer replaced by its
        chirality-inverted counterpart. The original molecule is not
        modified.

    Raises
    ------
    ValueError
        If the molecule does not have the requested conformer.

    Notes
    -----
    - Only the coordinates of the specified conformer are modified.
      Other conformers, if present, remain unchanged.
    - Chiral tags are reassigned from the updated 3D structure using
      :func:`rdkit.Chem.AssignAtomChiralTagsFromStructure` with
      ``replaceExistingTags=True``.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Input molecule has no conformers to invert.")

    try:
        conf_old = mol.GetConformer(conf_id)
    except ValueError as e:  # RDKit raises ValueError on invalid ID
        raise ValueError(
            f"Conformer with ID {conf_id} does not exist in the molecule."
        ) from e

    # Work on a copy so that the original molecule is not modified
    mol_inv = Chem.Mol(mol)

    conf_id = conf_old.GetId()
    if 0 <= conf_id < mol_inv.GetNumConformers():
        # Remove conformer with the same ID if it exists on the copy
        mol_inv.RemoveConformer(conf_id)

    coords = conf_old.GetPositions()
    inverted = invert_chirality_coords(coords)

    # Create a new conformer with inverted coordinates
    conf_new = Chem.Conformer(conf_old.GetNumAtoms())
    conf_new.SetId(conf_id)
    set_positions(conf_new, inverted)

    # Add the new conformer back to the copied molecule
    mol_inv.AddConformer(conf_new)

    # Recalculate chirality from coordinates
    Chem.AssignAtomChiralTagsFromStructure(mol_inv, replaceExistingTags=True)

    return mol_inv

