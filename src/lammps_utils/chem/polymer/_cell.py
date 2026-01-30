"""Module for generating amorphous cells using packmol."""

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from rdkit import Chem

from lammps_utils.helpers import calculate_box_length
from lammps_utils.logging import get_child_logger

logger = get_child_logger(__name__)

BIN_PACKMOL = os.environ.get("BIN_PACKMOL", "packmol")
"""Path to the packmol executable. Default is "packmol"."""


def generate_amorphous_cell(
    mol_and_numchains: Sequence[tuple[Chem.rdchem.Mol, int]],
    density: float = 0.3,
    tolerance: float = 2.0,
    nloop: int = 50,
    maxit: int = 20,
    seed: Optional[int] = None,
) -> Chem.rdchem.Mol:
    """
    Generate an amorphous cell using packmol.

    This function creates a periodic simulation cell containing multiple
    polymer chains at the specified density using the packmol program.

    Parameters
    ----------
    mol_and_numchains : Sequence[tuple[Chem.rdchem.Mol, int]]
        Tuple of (molecule, number_of_chains) pairs. Each molecule represents
        a polymer chain to be placed in the cell.
    density : float, optional
        Target density in g/cm³. Default is 0.3.
    tolerance : float, optional
        Distance tolerance for packmol in Angstroms. Default is 2.0.
    nloop : int, optional
        Number of loops for packmol optimization. Default is 50.
    maxit : int, optional
        Maximum number of iterations for packmol. Default is 20.
    seed : int, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    Chem.rdchem.Mol
        The generated amorphous cell as an RDKit molecule with all chains
        and their atom properties preserved.

    Raises
    ------
    subprocess.CalledProcessError
        If packmol execution fails.
    FileNotFoundError
        If packmol executable is not found.

    Notes
    -----
    This function:
    1. Calculates the box size from total mass and density
    2. Creates temporary PDB files for each polymer type
    3. Generates a packmol input file
    4. Executes packmol to pack molecules into the cell
    5. Reads the output and preserves atom properties

    The function uses a cubic periodic boundary condition (PBC) box.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from lammps_utils.chem.polymer import polymerize_linear
    >>> polymer = polymerize_linear((monomer,), (1.0,), n=10)
    >>> cell = generate_amorphous_cell(((polymer, 5),), density=0.3)
    """
    # Calculate box length from total mass and density
    total_mass = sum(
        sum(atom.GetMass() for atom in mol.GetAtoms()) * n_chain
        for mol, n_chain in mol_and_numchains
    )
    box_length = calculate_box_length(total_mass, density)
    box_length_half = box_length / 2

    box_cell_str = " ".join(
        map(str, (-box_length_half,) * 3 + (box_length_half,) * 3)
    )

    seed = seed if seed is not None else -1

    # Generate amorphous cell using packmol
    with tempfile.TemporaryDirectory() as str_dirpath_work:
        dirpath_work = Path(str_dirpath_work).resolve()

        filepath_packmol_input = dirpath_work / "packmol.inp"
        filepath_output = dirpath_work / "out.pdb"

        # Create packmol input file
        # Header section
        packmol_header = "\n".join(
            [
                f"tolerance  {tolerance}",
                "filetype  pdb",
                f"nloop  {nloop}",
                f"maxit  {maxit}",
                f"output  {filepath_output}",
                f"pbc  {box_cell_str}",
                f"seed  {seed}",
                "",
                "",
                "",
            ]
        )
        content_packmol_input = packmol_header

        # Structure section
        for i, (mol, n_chain) in enumerate(mol_and_numchains):
            filename_input_pdb = f"polymer_{i}.pdb"
            filepath_input_pdb = dirpath_work / filename_input_pdb

            Chem.MolToPDBFile(mol, str(filepath_input_pdb))

            content_packmol_input += "\n".join(
                [
                    f"structure  {filepath_input_pdb}",
                    f"    number  {n_chain}",
                    f"    inside box  {box_cell_str}",
                    "end structure",
                    "",
                    "",
                ]
            )

        # Write packmol input file
        filepath_packmol_input.write_text(content_packmol_input)

        # Execute packmol
        list_commands = [BIN_PACKMOL, "-i", str(filepath_packmol_input)]
        result_packmol = subprocess.run(
            list_commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        logger.debug(" ".join(list_commands))
        logger.debug(result_packmol.stdout.decode())
        logger.debug(result_packmol.stderr.decode())

        # Read the generated cell
        mol_ac = Chem.MolFromPDBFile(
            str(filepath_output), sanitize=False, removeHs=False
        )

    # Assign atom properties from original molecules
    atom_index_amorphous_cell: int = 0
    for mol, n_chain in mol_and_numchains:
        for _ in range(n_chain):
            for atom in mol.GetAtoms():
                assert isinstance(atom, Chem.rdchem.Atom)
                for key in atom.GetPropNames():
                    mol_ac.GetAtomWithIdx(atom_index_amorphous_cell).SetProp(
                        key, atom.GetProp(key)
                    )
                atom_index_amorphous_cell += 1

    return mol_ac
