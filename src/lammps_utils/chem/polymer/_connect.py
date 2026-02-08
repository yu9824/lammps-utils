"""Module for connecting molecules to build polymer structures."""

from array import array
from typing import Literal, Optional, overload

import numpy as np
from rdkit import Chem

from lammps_utils.chem.conformer._generate import minimize_conformer
from lammps_utils.chem.conformer._rotate import rotate_around_bond
from lammps_utils.io.mol import set_positions
from lammps_utils.logging import get_child_logger

logger = get_child_logger(__name__)


def has_tritium(mol: Chem.rdchem.Mol) -> bool:
    for atom in mol.GetAtoms():
        assert isinstance(atom, Chem.rdchem.Atom)
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 3:
            return True
    return False


def has_asterisk(mol: Chem.rdchem.Mol) -> bool:
    for atom in mol.GetAtoms():
        assert isinstance(atom, Chem.rdchem.Atom)
        if atom.GetAtomicNum() == 0:
            return True
    return False


def replace_to_tritium_marker(
    mol: Chem.rdchem.Mol, check_no_tritium: bool = True
) -> Chem.rdchem.Mol:
    if check_no_tritium and has_tritium(mol):
        raise ValueError("Tritium atom found")
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        assert isinstance(atom, Chem.rdchem.Atom)
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetIsotope(3)
    return mol


@overload
def get_head_and_tail(
    mol: Chem.rdchem.Mol,
    raise_not_unique: Literal[True] = True,
    raise_no_head_or_tail: Literal[True] = True,
) -> tuple[tuple[int], tuple[int]]: ...


@overload
def get_head_and_tail(
    mol: Chem.rdchem.Mol,
    raise_not_unique: Literal[False] = False,
    raise_no_head_or_tail: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]]: ...


def get_head_and_tail(
    mol: Chem.rdchem.Mol,
    raise_not_unique: bool = True,
    raise_no_head_or_tail: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Get the head and tail atoms in an RDKit molecule.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        An RDKit Mol object representing the molecule to analyze.
    raise_not_unique : bool, optional
        If True, raise a ValueError if more than one atom is found with a "head" or "tail" marker.
    raise_no_head_or_tail : bool, optional
        If True, raise a ValueError if the head or tail atom cannot be identified.

    Returns
    -------
    tuple[tuple[int, ...], tuple[int, ...]]
        A tuple (head_idx, tail_idx) containing the atom indices of the detected head and tail.
    """

    arr_head_indexes: "array[int]" = array("I")
    arr_tail_indexes: "array[int]" = array("I")
    for atom in mol.GetAtoms():
        assert isinstance(atom, Chem.rdchem.Atom)
        props = atom.GetPropsAsDict()
        if props.get("head", False):
            if len(arr_head_indexes) > 1 and raise_not_unique:
                raise ValueError("Multiple atoms marked as head")
            arr_head_indexes.append(atom.GetIdx())
        if props.get("tail", False):
            if len(arr_tail_indexes) > 1 and raise_not_unique:
                raise ValueError("Multiple atoms marked as tail")
            arr_tail_indexes.append(atom.GetIdx())
    if len(arr_head_indexes) == 0 and raise_no_head_or_tail:
        raise ValueError("No head atom found")
    elif len(arr_tail_indexes) == 0 and raise_no_head_or_tail:
        raise ValueError("No tail atom found")

    return tuple(arr_head_indexes), tuple(arr_tail_indexes)


@overload
def detect_head_and_tail(
    mol: Chem.rdchem.Mol,
    raise_not_unique: Literal[True] = True,
) -> tuple[tuple[int], tuple[int]]: ...


@overload
def detect_head_and_tail(
    mol: Chem.rdchem.Mol,
    raise_not_unique: Literal[False] = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]: ...


def detect_head_and_tail(
    mol: Chem.rdchem.Mol,
    raise_not_unique: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Identify the head and tail atoms in an RDKit molecule for polymerization.

    If there are exactly two labeled atoms, the atom with the lower index is assigned as the head,
    and the one with the higher index as the tail.

    Otherwise, all are considered both heads and tails.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        An RDKit Mol object representing the molecule to analyze.
    raise_not_unique : bool, optional
        If True, raise a ValueError if more than one atom is found with a "head" or "tail" marker.
    raise_no_head_or_tail : bool, optional
        If True, raise a ValueError if the head or tail atom cannot be identified.

    Returns
    -------
    tuple[int, int]
        A tuple (head_idx, tail_idx) containing the atom indices of the detected head and tail.

    Raises
    ------
    ValueError
        If more than one atom is found with a "head" or "tail" marker.
    AssertionError
        If either the head or tail atom cannot be identified.

    Notes
    -----
    - This function first inspects all atoms in the molecule for boolean properties "head" and "tail".
      If both are found, those indices are returned.
    - If such properties are not set, the function looks for hydrogen atoms with isotope 3 ([3H]):
        * The first [3H] atom encountered is assigned as the head, and its "head" property is set to True.
        * The second [3H] atom encountered is assigned as the tail, and its "tail" property is set to True.
    - If only one labeled ([3H]) atom is present, both head and tail are set to the same atom,
      and a warning is issued.
    - This detection mechanism supports both property-based and isotope-based conventions.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("[3H]CC(c1ccccc1)[3H]")
    >>> head_idx, tail_idx = detect_head_and_tail(mol)
    """
    arr_indexes: "array[int]" = array("I")
    for atom in mol.GetAtoms():
        assert isinstance(atom, Chem.rdchem.Atom)
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 3:
            arr_indexes.append(atom.GetIdx())
    if len(arr_indexes) == 2:
        head_idx = arr_indexes[0]
        tail_idx = arr_indexes[1]
        mol.GetAtomWithIdx(head_idx).SetBoolProp("head", True)
        mol.GetAtomWithIdx(tail_idx).SetBoolProp("tail", True)
        return ((head_idx,), (tail_idx,))
    elif raise_not_unique:
        if len(arr_indexes) == 0:
            raise ValueError("No head or tail atom found")
        elif len(arr_indexes) == 1:
            raise ValueError("Only one head or tail atom found")
        else:
            raise ValueError("Multiple head or tail atoms found")
    else:
        # logger.warning("")
        for idx in arr_indexes:
            mol.GetAtomWithIdx(idx).SetBoolProp("head", True)
            mol.GetAtomWithIdx(idx).SetBoolProp("tail", True)
        return tuple(arr_indexes), tuple(arr_indexes)


def rotation_matrix_from_vectors(
    source_vector: np.ndarray,
    reference_vector: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute a 3x3 rotation matrix that rotates a source vector to either
    a parallel direction of a reference vector.

    This function uses Rodrigues' rotation formula:

        R = I + sin(theta) * K + (1 - cos(theta)) * K^2

    where K is the skew-symmetric matrix of the rotation axis.

    Parameters
    ----------
    source_vector : np.ndarray
        Vector to be rotated, shape (3,).
    reference_vector : np.ndarray
        Reference direction vector, shape (3,).
    eps : float
        Numerical tolerance.

    Returns
    -------
    np.ndarray
        Proper rotation matrix (3, 3) with det(R)=+1.
    """
    source_unit = np.asarray(source_vector, dtype=float)
    target_unit = np.asarray(reference_vector, dtype=float)

    source_unit /= np.linalg.norm(source_unit)
    target_unit /= np.linalg.norm(target_unit)

    cos_theta = np.dot(source_unit, target_unit)

    if cos_theta > 1.0 - eps:
        return np.eye(3)

    if cos_theta < -1.0 + eps:
        # 180-degree rotation
        fallback_axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(source_unit, fallback_axis)) > 0.9:
            fallback_axis = np.array([0.0, 1.0, 0.0])

        rotation_axis = np.cross(source_unit, fallback_axis)
        rotation_axis /= np.linalg.norm(rotation_axis)

        K = np.array(
            [
                [0.0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0.0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0.0],
            ]
        )

        return np.eye(3) + 2.0 * (K @ K)

    # General Rodrigues rotation
    rotation_axis = np.cross(source_unit, target_unit)
    sin_theta = np.linalg.norm(rotation_axis)
    rotation_axis /= sin_theta

    K = np.array(
        [
            [0.0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0.0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0.0],
        ]
    )

    return (
        np.eye(3)
        + np.sin(np.arccos(cos_theta)) * K
        + (1.0 - cos_theta) * (K @ K)
    )


# TODO: 結合角の指定
def connect_mols(
    mol1: Chem.rdchem.Mol,
    mol2: Chem.rdchem.Mol,
    idx1: int,
    idx2: int,
    bond_length: float = 1.5,
    bond_type: Chem.BondType = Chem.BondType.SINGLE,
    random_walk: bool = False,
    torsion_angle: Optional[float] = None,
    align_conformer: bool = True,
    forcefield: Optional[Literal["MMFF", "UFF"]] = None,
    seed: Optional[int] = None,
) -> Chem.rdchem.Mol:
    """
    Connect two molecules by removing dummy atoms (idx1, idx2) and forming
    a bond between their respective neighbor atoms with a well-defined
    geometric alignment.

    Geometry rule
    -------------
    Let:

        vec1 = idx1 -> target_idx1   (mol1 side)
        vec2 = idx2 -> target_idx2   (mol2 side)

    Then mol2 is rigidly transformed so that:

        vec2 aligns with -vec1

    and target_idx2 is placed at:

        target_idx1 + vec1 * bond_length

    Notes
    -----
    - idx1 and idx2 are assumed to be dummy atoms (e.g. Tritium).
    - Both idx1 and idx2 must have exactly one neighbor.
    - rotate_around_bond is intentionally NOT applied here.
    """
    rng = np.random.default_rng(seed)

    mol1 = Chem.Mol(mol1)
    mol2 = Chem.Mol(mol2)

    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()

    atom1 = mol1.GetAtomWithIdx(idx1)
    atom2 = mol2.GetAtomWithIdx(idx2)

    assert atom1.GetDegree() == 1
    assert atom2.GetDegree() == 1

    target_idx1 = atom1.GetNeighbors()[0].GetIdx()
    target_idx2 = atom2.GetNeighbors()[0].GetIdx()

    # --- direction vectors ---
    origin1 = np.asarray(conf1.GetAtomPosition(target_idx1))
    origin2 = np.asarray(conf2.GetAtomPosition(target_idx2))

    vec1 = np.asarray(conf1.GetAtomPosition(idx1)) - origin1
    vec1 /= np.linalg.norm(vec1)
    vec2 = np.asarray(conf2.GetAtomPosition(idx2)) - origin2
    vec2 /= np.linalg.norm(vec2)

    if random_walk:
        walk_vec = rng.uniform(-1.0, 1.0, 3)
        walk_vec /= np.linalg.norm(walk_vec)
    else:
        walk_vec = vec1

    if align_conformer:
        # R = rotation_matrix_from_vectors(vec2, -vec1)
        R = rotation_matrix_from_vectors(vec2, -walk_vec)
    else:
        R = np.eye(3)
    pos2_rot = (R @ (conf2.GetPositions() - origin2).T).T + origin2

    # --- translate mol2 to satisfy bond length ---
    shift = origin1 - origin2 + bond_length * walk_vec
    pos2_final = pos2_rot + shift

    # write back coordinates
    if torsion_angle is not None:
        pos2_final = rotate_around_bond(
            pos2_final, target_idx2, idx2, torsion_angle
        )
    set_positions(conf2, pos2_final)

    # --- combine molecules ---
    combo = Chem.CombineMols(mol1, mol2)
    rw = Chem.RWMol(combo)

    offset = mol1.GetNumAtoms()
    rw.AddBond(target_idx1, target_idx2 + offset, bond_type)

    # remove dummy atoms
    rw.RemoveAtom(idx2 + offset)
    rw.RemoveAtom(idx1)

    connected = rw.GetMol()
    Chem.SanitizeMol(connected)

    if forcefield is not None:
        minimize_conformer(connected, forcefield=forcefield)

    return connected
