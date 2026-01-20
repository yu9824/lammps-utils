"""Module for connecting molecules to build polymer structures."""

from typing import Literal, Optional

import numpy as np
from rdkit import Chem

from lammps_utils.chem.conformer._generate import minimize_conformer
from lammps_utils.chem.conformer._rotate import rotate_around_bond
from lammps_utils.logging import get_child_logger

logger = get_child_logger(__name__)


def detect_head_and_tail(mol: Chem.rdchem.Mol) -> tuple[int, int]:
    """
    Identify the head and tail atoms in an RDKit molecule for polymerization.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        An RDKit Mol object representing the molecule to analyze.

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
    head_idx: int = -1
    tail_idx: int = -1
    for atom in mol.GetAtoms():
        assert isinstance(atom, Chem.rdchem.Atom)
        props = atom.GetPropsAsDict()
        if props.get("head", False):
            if head_idx >= 0:
                raise ValueError("Multiple atoms marked as head")
            head_idx = atom.GetIdx()
        elif props.get("tail", False):
            if tail_idx >= 0:
                raise ValueError("Multiple atoms marked as tail")
            tail_idx = atom.GetIdx()

    if head_idx >= 0 and tail_idx >= 0:
        return head_idx, tail_idx

    for atom in mol.GetAtoms():
        assert isinstance(atom, Chem.rdchem.Atom)
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 3:
            if head_idx < 0:
                head_idx = atom.GetIdx()
                atom.SetBoolProp("head", True)
            else:
                tail_idx = atom.GetIdx()
                atom.SetBoolProp("tail", True)
                break
    else:
        if head_idx >= 0 and tail_idx < 0:
            logger.warning(
                "Only head_idx found and not tail_idx. "
                "Setting tail_idx to head_idx. "
                "Please check the atom properties for 'tail' marker."
            )
            tail_idx = head_idx

    assert head_idx >= 0, "head_idx not found"
    assert tail_idx >= 0, "tail_idx not found"
    return head_idx, tail_idx


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

    conf1.SetPositions(
        conf1.GetPositions()
        - np.asarray(conf1.GetAtomPosition(target_idx1))
        + rand_vec
    )
    conf2.SetPositions(
        conf2.GetPositions() - np.asarray(conf2.GetAtomPosition(target_idx2))
    )
    if angle is None:
        angle = rng.uniform(0, 2 * np.pi)
    conf2.SetPositions(
        rotate_around_bond(conf2.GetPositions(), idx2, target_idx2, angle)
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
