from __future__ import annotations

from array import array

import numpy as np
from rdkit import Chem

from lammps_utils.chem.polymer._connect import resolve_head_and_tail
from lammps_utils.logging import get_child_logger

logger = get_child_logger(__name__)


def improper_handedness(
    mol: Chem.Mol,
    atom_prev: int,
    atom_center: int,
    atom_next: int,
    atom_substituent: int,
    conf_id: int = -1,
) -> int:
    """
    Compute the handedness (chirality sign) of a tetrahedral center
    using an improper dihedral defined by four atoms.

    The handedness is determined from the scalar triple product:

        sign = ( (r_next - r_center) × (r_prev - r_center) )
               · (r_substituent - r_center)

    where:
        - atom_center      : central (chiral) atom
        - atom_prev        : one backbone neighbor (e.g., previous in main chain)
        - atom_next        : the other backbone neighbor (e.g., next in main chain)
        - atom_substituent : side-chain atom attached to the center

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule containing a 3D conformer.
    atom_prev : int
        Atom index of the previous backbone atom.
    atom_center : int
        Atom index of the chiral center.
    atom_next : int
        Atom index of the next backbone atom.
    atom_substituent : int
        Atom index of the side-chain atom.
    conf_id : int, optional
        Conformer ID to use. Default is -1 (last conformer).

    Returns
    -------
    int
        +1  : right-handed orientation
        -1  : left-handed orientation
         0  : nearly planar (degenerate case, |value| < 1e-8)

    Notes
    -----
    - The sign depends on the order of (atom_next, atom_prev).
      Swapping them flips the result.
    - This function does NOT depend on RDKit's CIP R/S assignment.
      It purely evaluates 3D geometry.
    - Suitable for tacticity evaluation in polymers when the
      backbone direction is consistently defined.
    """

    conf = mol.GetConformer(conf_id)

    def position(atom_index: int) -> np.ndarray:
        p = conf.GetAtomPosition(atom_index)
        return np.array([p.x, p.y, p.z], dtype=float)

    # Vectors from central atom
    vec_next = position(atom_next) - position(atom_center)
    vec_prev = position(atom_prev) - position(atom_center)
    vec_sub = position(atom_substituent) - position(atom_center)

    # Scalar triple product
    signed_volume = float(np.dot(np.cross(vec_next, vec_prev), vec_sub))

    eps = 1e-8
    if signed_volume > eps:
        return 1
    elif signed_volume < -eps:
        return -1
    else:
        return 0


def compute_main_chain_handedness(
    mol_poly: Chem.rdchem.Mol,
    conf_id: int = -1,
) -> tuple[int, ...]:
    """
    Get the handedness of each tetrahedral chiral center along the main chain.

    The main chain is taken as the shortest path between head and tail atoms
    (see :func:`lammps_utils.chem.polymer._connect.resolve_head_and_tail`).
    For each chiral center on the chain (excluding endpoints), the 3D handedness
    is computed from the improper dihedral (backbone–center–forward, sidechain)
    via :func:`improper_handedness`. When a center has multiple sidechain atoms,
    the one with the largest mass is used as the reference; ties raise.

    Parameters
    ----------
    mol_poly : Chem.rdchem.Mol
        RDKit molecule (polymer) with at least one conformer.
    conf_id : int, optional
        Conformer ID used for 3D coordinates. Default is -1.

    Returns
    -------
    tuple[int, ...]
        Handedness for each chiral center in main-chain order. Each element is
        +1 (counterclockwise), -1 (clockwise), or 0 (coplanar); see
        :func:`improper_handedness`.

    Raises
    ------
    ValueError
        If a chiral center has two sidechain atoms with the same mass, so the
        reference sidechain cannot be chosen uniquely.
    """
    head_indexes, tail_indexes = resolve_head_and_tail(mol_poly)
    atom_indexes_main_chain: tuple[int, ...] = tuple(
        Chem.GetShortestPath(mol_poly, head_indexes[0], tail_indexes[0])
    )
    set_atom_indexes_main_chain = set(atom_indexes_main_chain)

    arr_handedness = array("i")
    for i in range(len(atom_indexes_main_chain)):
        idx_center = atom_indexes_main_chain[i]
        logger.debug(f"idx_center: {idx_center}")
        atom_center = mol_poly.GetAtomWithIdx(idx_center)

        if atom_center.GetChiralTag() not in {
            Chem.CHI_TETRAHEDRAL_CCW,
            Chem.CHI_TETRAHEDRAL_CW,
        }:
            continue

        idx_backbone = atom_indexes_main_chain[i - 1]
        idx_forward = atom_indexes_main_chain[i + 1]

        atom_sidechains: tuple[Chem.rdchem.Atom, ...] = tuple(
            n
            for n in atom_center.GetNeighbors()
            if n.GetIdx() not in set_atom_indexes_main_chain
        )

        if not atom_sidechains:
            raise ValueError("no sidechain atoms found")

        if len(atom_sidechains) == 1:
            atom_sidechain = atom_sidechains[0]
        else:
            atom_sidechains = tuple(
                sorted(
                    atom_sidechains,
                    key=lambda a: (a.GetMass(), a.GetIsotope()),
                    reverse=True,
                )
            )
            if (
                atom_sidechains[0].GetMass() == atom_sidechains[1].GetMass()
                and atom_sidechains[0].GetIsotope()
                == atom_sidechains[1].GetIsotope()
            ):
                raise ValueError("sidechain mass and isotope are the same")
            atom_sidechain = atom_sidechains[0]

        arr_handedness.append(
            improper_handedness(
                mol_poly,
                idx_backbone,
                idx_center,
                idx_forward,
                atom_sidechain.GetIdx(),
                conf_id=conf_id,
            )
        )
    return tuple(arr_handedness)
