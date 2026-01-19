"""Module for connecting molecules to build polymer structures."""

from typing import Literal, Optional

import numpy as np
from rdkit import Chem

from lammps_utils.chem.conformer._generate import minimize_conformer
from lammps_utils.chem.conformer._rotate import rotate_around_bond


def detect_head_and_tail(mol: Chem.rdchem.Mol) -> tuple[int, int]:
    """
    Detect head and tail atoms in a molecule for polymerization.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        RDKit Mol object to analyze.

    Returns
    -------
    tuple[int, int]
        A tuple containing (head_idx, tail_idx) atom indices.

    Raises
    ------
    ValueError
        If multiple atoms are marked as head_idx or tail_idx.
    AssertionError
        If head_idx or tail_idx cannot be found.

    Notes
    -----
    This function first checks for atoms with "head_idx" or "tail_idx" boolean properties.
    If not found, it searches for hydrogen atoms with isotope number 3 ([3H]).
    The first [3H] atom found is marked as head, and the second as tail.

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
        if props.get("head_idx", False):
            if head_idx >= 0:
                raise ValueError("Multiple atoms marked as head_idx")
            head_idx = atom.GetIdx()
        elif props.get("tail_idx", False):
            if tail_idx >= 0:
                raise ValueError("Multiple atoms marked as tail_idx")
            tail_idx = atom.GetIdx()

    if head_idx >= 0 and tail_idx >= 0:
        return head_idx, tail_idx

    for atom in mol.GetAtoms():
        assert isinstance(atom, Chem.rdchem.Atom)
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 3:
            if head_idx < 0:
                head_idx = atom.GetIdx()
                atom.SetBoolProp("head_idx", True)
            else:
                tail_idx = atom.GetIdx()
                atom.SetBoolProp("tail_idx", True)
                break

    assert head_idx >= 0, "head_idx not found"
    assert tail_idx >= 0, "tail_idx not found"
    return head_idx, tail_idx


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

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol1 = Chem.MolFromSmiles("CCO")
    >>> mol2 = Chem.MolFromSmiles("CCO")
    >>> connected = connect_mols(mol1, mol2, 0, 0, angle=0.0)
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
