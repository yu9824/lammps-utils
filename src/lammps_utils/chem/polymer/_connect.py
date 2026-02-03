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
        elif props.get("tail", False):
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


# TODO: 結合角の指定
def connect_mols(
    mol1: Chem.rdchem.Mol,
    mol2: Chem.rdchem.Mol,
    idx1: int,
    idx2: int,
    bond_length: float = 1.5,
    bond_type: Chem.BondType = Chem.BondType.SINGLE,
    angle: Optional[float] = None,
    seed: Optional[int] = None,
    forcefield: Optional[Literal["MMFF", "UFF"]] = "MMFF",
) -> Chem.rdchem.Mol:
    """
    Connect two molecules by forming a bond between specified atoms.

    Parameters
    ----------
    mol1 : Chem.rdchem.Mol
        First molecule to connect.
    mol2 : Chem.rdchem.Mol
        Second molecule to connect.
    idx1 : int
        Index of the atom in mol1 to connect (must have exactly one neighbor).
    idx2 : int
        Index of the atom in mol2 to connect (must have exactly one neighbor).
    bond_length : float, optional
        Desired bond length in Angstroms. Default is 1.5.
    bond_type : Chem.BondType, optional
        Type of bond to form. Default is Chem.BondType.SINGLE.
    angle : Optional[float], optional
        Rotation angle in radians around the bond. If None, a random angle is used.
        Default is None.
    seed : Optional[int], optional
        Random seed for generating random rotation angle and vector. Default is None.
    forcefield : Optional[Literal["MMFF", "UFF"]], optional
        Force field to use for energy minimization after connection.
        If None, no minimization is performed. Default is "MMFF".

    Returns
    -------
    Chem.rdchem.Mol
        The connected molecule with sanitized structure.

    Raises
    ------
    AssertionError
        If idx1 or idx2 atoms do not have exactly one neighbor.

    Notes
    -----
    This function:
    1. Creates copies of the input molecules
    2. Positions mol2 relative to mol1 using a random vector
    3. Optionally rotates mol2 around the bond axis
    4. Combines the molecules and forms a bond between the target atoms
    5. Removes the connecting atoms (idx1 and idx2)
    6. Sanitizes the resulting molecule
    7. Optionally minimizes the conformer energy

    """
    rng = np.random.default_rng(seed=seed)

    mol1 = Chem.Mol(mol1)
    mol2 = Chem.Mol(mol2)

    rand_vec = rng.normal(size=3)
    rand_vec = rand_vec / np.linalg.norm(rand_vec) * bond_length

    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()

    atom1 = mol1.GetAtomWithIdx(idx1)
    assert len(atom1.GetNeighbors()) == 1

    atom2 = mol2.GetAtomWithIdx(idx2)
    assert len(atom2.GetNeighbors()) == 1

    target_atom1 = atom1.GetNeighbors()[0]
    target_atom2 = atom2.GetNeighbors()[0]
    assert isinstance(target_atom1, Chem.rdchem.Atom)
    assert isinstance(target_atom2, Chem.rdchem.Atom)

    target_idx1 = target_atom1.GetIdx()
    target_idx2 = target_atom2.GetIdx()

    set_positions(
        conf1,
        conf1.GetPositions()
        - np.asarray(conf1.GetAtomPosition(target_idx1))
        + rand_vec,
    )

    set_positions(
        conf2,
        conf2.GetPositions() - np.asarray(conf2.GetAtomPosition(target_idx2)),
    )

    if angle is None:
        angle = rng.uniform(0, 2 * np.pi)

    set_positions(
        conf2,
        rotate_around_bond(conf2.GetPositions(), idx2, target_idx2, angle),
    )

    combo = Chem.CombineMols(mol1, mol2)

    rw = Chem.RWMol(combo)
    offset = mol1.GetNumAtoms()
    rw.AddBond(target_idx1, target_idx2 + offset, bond_type)
    rw.RemoveAtom(idx2 + offset)
    rw.RemoveAtom(idx1)

    connected_mol = rw.GetMol()
    Chem.SanitizeMol(connected_mol)

    if forcefield:
        minimize_conformer(connected_mol, forcefield=forcefield)
    return connected_mol
