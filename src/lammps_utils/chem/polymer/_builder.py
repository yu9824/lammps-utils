"""Module for building polymer structures from monomers."""

from typing import Literal, Optional

import numpy as np
from rdkit import Chem

from lammps_utils.chem.conformer._generate import minimize_conformer
from lammps_utils.chem.polymer._connect import (
    connect_mols,
    detect_head_and_tail,
    get_head_and_tail,
)


def polymerize_linear(
    mols: tuple[Chem.rdchem.Mol, ...],
    ratio: tuple[float, ...] = (1.0,),
    n: int = 10,
    forcefield: Optional[Literal["MMFF", "UFF"]] = "MMFF",
    seed: Optional[int] = None,
) -> Chem.rdchem.Mol:
    """
    Create a linear polymer by connecting multiple monomer molecules.

    Parameters
    ----------
    mols : tuple[Chem.rdchem.Mol, ...]
        Tuple of monomer molecules to polymerize. Each molecule must have
        detectable head and tail atoms (e.g., [3H] markers).
    ratio : tuple[float, ...], optional
        Relative ratio for selecting each monomer type. Must have the same
        length as mols. Default is (1.0,) for a single monomer type.
    n : int, optional
        Number of monomer units in the polymer chain. Default is 10.
    forcefield : Optional[Literal["MMFF", "UFF"]], optional
        Force field to use for energy minimization after polymerization.
        If None, no minimization is performed. Default is "MMFF".
    seed : Optional[int], optional
        Random seed for monomer selection and connection angles.
        Default is None.

    Returns
    -------
    Chem.rdchem.Mol
        The resulting linear polymer molecule.

    Raises
    ------
    AssertionError
        If the length of mols and ratio do not match, or if monomers
        do not have detectable head and tail atoms.

    Notes
    -----
    This function:
    1. Randomly selects n monomers from the input set according to the ratio
    2. Connects them sequentially: tail of previous -> head of next
    3. Uses detect_head_and_tail to find connection points automatically
    4. Optionally minimizes the final polymer structure

    Examples
    --------
    >>> from rdkit import Chem
    >>> from lammps_utils.chem.conformer import generate_minimized_conformer
    >>> monomer = generate_minimized_conformer("[3H]CC(c1ccccc1)[3H]")
    >>> polymer = polymerize_linear((monomer,), (1.0,), n=5, seed=42)
    """
    assert len(mols) == len(ratio), "Length of mols and ratio must match"

    rng = np.random.default_rng(seed)
    selected_mols = rng.choice(mols, size=n, replace=True, p=ratio).tolist()

    mol = selected_mols[0]
    for i in range(1, n):
        mol2 = selected_mols[i]

        head_indexes, tail_indexes = get_head_and_tail(
            mol, raise_no_head_or_tail=False
        )
        head_indexes2, tail_indexes2 = get_head_and_tail(
            mol2, raise_no_head_or_tail=False
        )
        if not (head_indexes or tail_indexes):
            head_indexes, tail_indexes = detect_head_and_tail(mol)
            head_indexes2, tail_indexes2 = detect_head_and_tail(mol2)

        # Connect tail of current polymer to head of next monomer
        # Skip minimization during intermediate steps for efficiency
        mol = connect_mols(
            mol, mol2, tail_indexes[0], head_indexes2[0], forcefield=None
        )

    # Final minimization of the complete polymer
    if forcefield:
        minimize_conformer(mol, forcefield=forcefield)
    return mol


def attach_terminal_groups(
    polymer: Chem.rdchem.Mol,
    terminal: Chem.rdchem.Mol,
    forcefield: Optional[Literal["MMFF", "UFF"]] = "MMFF",
) -> Chem.rdchem.Mol:
    """
    Attach terminal groups to both ends of a linear polymer.

    Parameters
    ----------
    polymer : Chem.rdchem.Mol
        The polymer molecule to which terminal groups will be attached.
        Must have detectable head and tail atoms (e.g., [3H] markers).
    terminal : Chem.rdchem.Mol
        The terminal group molecule (e.g., CH3) to attach at both ends.
        Must have detectable head and tail atoms.
    forcefield : Optional[Literal["MMFF", "UFF"]], optional
        Force field to use for energy minimization after attachment.
        If None, no minimization is performed. Default is "MMFF".

    Returns
    -------
    Chem.rdchem.Mol
        The polymer molecule with terminal groups attached at both ends.

    Notes
    -----
    This function:
    1. Detects head and tail atoms of both polymer and terminal group
    2. Connects terminal group to the tail end of the polymer
    3. Connects another terminal group to the head end of the polymer
    4. Optionally minimizes the final structure

    The head and tail atoms are detected automatically using detect_head_and_tail,
    which looks for [3H] markers or atom properties.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from lammps_utils.chem.conformer import generate_minimized_conformer
    >>> polymer = generate_minimized_conformer("[3H]CC(c1ccccc1)[3H]")
    >>> terminal = generate_minimized_conformer("[3H]C[3H]")
    >>> capped = attach_terminal_groups(polymer, terminal)
    """

    mol = Chem.Mol(polymer)

    head_indexes_terminal, tail_indexes_terminal = get_head_and_tail(
        terminal, raise_no_head_or_tail=False, raise_not_unique=False
    )
    head_indexes_polymer, tail_indexes_polymer = get_head_and_tail(
        mol, raise_no_head_or_tail=False, raise_not_unique=False
    )
    if not (head_indexes_terminal or tail_indexes_terminal):
        head_indexes_terminal, tail_indexes_terminal = detect_head_and_tail(
            terminal, raise_not_unique=False
        )
    if not (head_indexes_polymer or tail_indexes_polymer):
        head_indexes_polymer, tail_indexes_polymer = detect_head_and_tail(
            mol, raise_not_unique=False
        )

    # Attach terminal group to tail end
    for tail_index_polymer in tail_indexes_polymer:
        mol = connect_mols(
            mol,
            terminal,
            tail_index_polymer,
            head_indexes_terminal[0],
            forcefield=None,
        )

    for head_index_polymer in head_indexes_polymer:
        mol = connect_mols(
            mol,
            terminal,
            head_index_polymer,
            tail_indexes_terminal[0],
            forcefield=None,
        )

    # Final minimization of the complete structure
    if forcefield:
        minimize_conformer(mol, forcefield=forcefield)
    return mol
