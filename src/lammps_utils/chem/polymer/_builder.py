"""Module for building polymer structures from monomers."""

from itertools import cycle, islice
from typing import Literal, Optional, Union

import numpy as np
from rdkit import Chem

from lammps_utils.chem.conformer._chirality import invert_chirality
from lammps_utils.chem.conformer._generate import minimize_conformer
from lammps_utils.chem.polymer._connect import (
    connect_mols,
    get_head_and_tail_from_props,
    infer_head_and_tail,
    resolve_head_and_tail,
)
from lammps_utils.chem.polymer._handedness import compute_main_chain_handedness
from lammps_utils.logging import get_child_logger

logger = get_child_logger(__name__)


def polymerize_linear(
    mols: tuple[Chem.rdchem.Mol, ...],
    ratio: tuple[float, ...] = (1.0,),
    n: int = 10,
    forcefield: Optional[Literal["MMFF", "UFF"]] = "MMFF",
    random_walk: bool = False,
    torsion_angle: Union[float, Literal["random"]] = "random",
    align_conformer: bool = True,
    tacticity: Optional[
        Literal["isotactic", "syndiotactic", "atactic"]
    ] = None,
    seed: Optional[int] = None,
) -> Chem.rdchem.Mol:
    """
    Create a linear polymer by connecting monomer molecules head-to-tail.

    Monomers are selected according to ``ratio``, connected in sequence
    (tail of current -> head of next), and the final structure is
    optionally minimized. When ``tacticity`` is set, main-chain handedness
    is checked after building (and after minimization if applicable).

    Parameters
    ----------
    mols : tuple[Chem.rdchem.Mol, ...]
        Monomer molecules to polymerize. Each must have detectable head
        and tail atoms (e.g., [3H] markers) and at least one conformer.
    ratio : tuple[float, ...], optional
        Relative ratio for selecting each monomer type. Length must match
        ``mols``. Default is (1.0,) for a single monomer type.
    n : int, optional
        Number of monomer units in the chain. Default is 10.
    forcefield : Optional[Literal["MMFF", "UFF"]], optional
        Force field for energy minimization of the final polymer. If None,
        no minimization is performed. Default is "MMFF".
    random_walk : bool, optional
        If True, use random-walk placement when connecting monomers.
        Incompatible with ``tacticity``. Default is False.
    torsion_angle : float or "random", optional
        Torsion angle (degrees) at the new bond, or "random" to sample
        randomly. Default is "random".
    align_conformer : bool, optional
        If True, align the incoming monomer to the growing chain before
        connection. Default is True.
    tacticity : Optional[Literal["isotactic", "syndiotactic", "atactic"]], optional
        Stereoregularity of the main-chain chiral centers. Only supported
        for a single monomer type (len(mols) == 1) with 1 or 2 chiral
        centers on the main chain. Ignored when multiple monomer types
        are given. Definitions:
        - isotactic: all chiral centers same handedness (+1 or -1).
        - syndiotactic: alternating handedness (+1, -1, +1, -1, ...).
          Not allowed with forcefield minimization.
        - atactic: random mix of +1 and -1.
        Default is None (no tacticity control or check).
    seed : Optional[int], optional
        Random seed for monomer selection and connection angles.
        Default is None.

    Returns
    -------
    Chem.rdchem.Mol
        Linear polymer molecule with one conformer.

    Raises
    ------
    ValueError
        If mols/ratio length mismatch, no conformers in a monomer,
        syndiotactic with forcefield minimization, tacticity with
        random_walk, or (when tacticity is set) tacticity check fails
        after building.
    AssertionError
        If head/tail resolution or internal tacticity assumptions fail.

    Notes
    -----
    1. Monomers are chosen according to ``ratio`` (random for isotactic/atactic/None,
       alternating for syndiotactic).
    2. Chains are built by repeated tail->head connection via ``connect_mols``
       without intermediate minimization.
    3. Optional final minimization is applied to the full polymer.
    4. When ``tacticity`` is not None, main-chain handedness is computed and
       asserted (isotactic: one unique value; syndiotactic: alternating odd/even;
       atactic: two values).

    Examples
    --------
    >>> from lammps_utils.chem.conformer import generate_minimized_conformer
    >>> monomer = generate_minimized_conformer("[3H]CC(c1ccccc1)[3H]")
    >>> polymer = polymerize_linear((monomer,), (1.0,), n=5, seed=42)
    >>> polymer = polymerize_linear((monomer,), (1.0,), n=5, tacticity="isotactic", seed=42)
    """
    assert len(mols) == len(ratio), "Length of mols and ratio must match"

    # raise errors when not supported combinations are used
    if tacticity == "syndiotactic" and forcefield is not None:
        # FIXME: forcefield minimizationすると tacticlityが崩れてしまう
        raise ValueError(
            "Syndiotactic tacticity is not supported with forcefield minimization."
        )
    if tacticity is not None and random_walk:
        raise ValueError("Tacticity is not supported with random walk.")

    # raise errors when no conformers are found
    for mol in mols:
        if mol.GetNumConformers() == 0:
            raise ValueError("No conformers found in the molecule.")

    if len(mols) != 1:
        tacticity = None
        logger.warning(
            "'tacticity' is ignored when multiple monomers are provided."
        )
    elif tacticity is not None:
        assert len(mols) == 1
        mol = mols[0]

        Chem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=True)
        indexes_head, indexes_tail = resolve_head_and_tail(mol)
        assert len(indexes_head) == 1
        assert len(indexes_tail) == 1
        indexes_main_chain: set[int] = set(
            Chem.GetShortestPath(mol, indexes_tail[0], indexes_head[0])
        )

        n_chiral_centers = 0
        for chiral_center, _ in Chem.FindMolChiralCenters(
            mol, includeUnassigned=False
        ):
            if chiral_center in indexes_main_chain:
                n_chiral_centers += 1
        if n_chiral_centers == 0:
            logger.warning(
                "No chiral centers found in the main chain. "
                "Turning off chiral tagging."
            )
            tacticity = None
        elif n_chiral_centers > 2:
            logger.warning(
                "More than 2 chiral centers found in the main chain. "
                "Turning off chiral tagging."
            )
            tacticity = None
        else:
            mol_inv = invert_chirality(mol)

            mols = (mol, mol_inv)
            ratio = (ratio[0] / 2, ratio[0] / 2)

    rng = np.random.default_rng(seed)

    selected_mols: tuple[Chem.Mol, ...]
    if tacticity in {"atactic", "isotactic", None}:
        selected_mols = tuple(
            rng.choice(mols, size=n, replace=True, p=ratio).tolist()
        )
        if tacticity == "atactic":
            assert len(mols) == 2
            assert len(ratio) == 2

    elif tacticity == "syndiotactic":
        assert len(mols) == 2
        assert len(ratio) == 2
        selected_mols = tuple(islice(cycle(mols), n))
    else:
        raise ValueError(f"{tacticity}")

    mol = selected_mols[0]
    for i in range(1, n):
        mol2 = selected_mols[i]

        head_indexes, tail_indexes = resolve_head_and_tail(mol)
        head_indexes2, tail_indexes2 = resolve_head_and_tail(mol2)

        # Connect tail of current polymer to head of next monomer
        # Skip minimization during intermediate steps for efficiency
        mol = connect_mols(
            mol,
            mol2,
            tail_indexes[0],
            head_indexes2[0],
            forcefield=None,
            random_walk=random_walk,
            torsion_angle=torsion_angle,
            align_conformer=align_conformer,
            seed=seed,
        )

    # Final minimization of the complete polymer
    if forcefield:
        minimize_conformer(mol, forcefield=forcefield)

    if tacticity is not None:
        handedness = compute_main_chain_handedness(mol)
        assert handedness, "tacticity is broken"
        if tacticity == "syndiotactic":
            handedness_odd = handedness[::2]
            handedness_even = handedness[1::2]
            assert len(set(handedness_odd)) == 1, "tacticity is broken"
            assert len(set(handedness_even)) == 1, "tacticity is broken"
            assert handedness_odd[0] != handedness_even[0], (
                "tacticity is broken"
            )
        elif tacticity == "isotactic":
            assert len(set(handedness)) == 1, "tacticity is broken"
        elif tacticity == "atactic":
            assert len(set(handedness)) == 2, "tacticity is broken"

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

    The head and tail atoms are resolved using get_head_and_tail_from_props
    and infer_head_and_tail (e.g. [3H] markers or atom properties).

    Examples
    --------
    >>> from rdkit import Chem
    >>> from lammps_utils.chem.conformer import generate_minimized_conformer
    >>> polymer = generate_minimized_conformer("[3H]CC(c1ccccc1)[3H]")
    >>> terminal = generate_minimized_conformer("[3H]C[3H]")
    >>> capped = attach_terminal_groups(polymer, terminal)
    """

    mol = Chem.Mol(polymer)

    head_indexes_terminal, tail_indexes_terminal = (
        get_head_and_tail_from_props(
            terminal, raise_no_head_or_tail=False, raise_not_unique=False
        )
    )
    head_indexes_polymer, tail_indexes_polymer = get_head_and_tail_from_props(
        mol, raise_no_head_or_tail=False, raise_not_unique=False
    )
    if not (head_indexes_terminal or tail_indexes_terminal):
        head_indexes_terminal, tail_indexes_terminal = infer_head_and_tail(
            terminal, raise_not_unique=False
        )
    if not (head_indexes_polymer or tail_indexes_polymer):
        head_indexes_polymer, tail_indexes_polymer = infer_head_and_tail(
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
            torsion_angle="random",
            align_conformer=True,
            random_walk=False,
        )

    for head_index_polymer in head_indexes_polymer:
        mol = connect_mols(
            mol,
            terminal,
            head_index_polymer,
            tail_indexes_terminal[0],
            forcefield=None,
            torsion_angle="random",
            align_conformer=True,
            random_walk=False,
        )

    # Final minimization of the complete structure
    if forcefield:
        minimize_conformer(mol, forcefield=forcefield)
    return mol
