"""Functions for converting MOL2 files to .lt format and generating LAMMPS input files using Moltemplate.

This module provides functions to generate .lt files using Moltemplate (https://moltemplate.org)
and to create LAMMPS data and input files from molecular structures.
"""

import importlib
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Union

from rdkit import Chem

from lammps_utils.chem.polymer import (
    calculate_box_length,
    generate_amorphous_cell,
)
from lammps_utils.helpers import run_executable, work_directory
from lammps_utils.io.mol import MolToMol2File
from lammps_utils.logging import get_child_logger
from lammps_utils.types import CellBounds

logger = get_child_logger(__name__)

# Type alias for mol_spec: (mol_name, filepath_lt, mol_count)
MolSpec = tuple[str, Union[os.PathLike, str], int]


def mol22lt(
    filepath_mol2: Union[str, os.PathLike],
    filepath_lt: Optional[Union[str, os.PathLike]] = None,
    forcefield: Literal["GAFF2"] = "GAFF2",
    name: str = "UNL",
) -> None:
    """Convert a MOL2 file to a Moltemplate (.lt) file via moltemplate's mol22lt.py.

    Requires moltemplate to be installed (pip install moltemplate).
    The mol22lt.py script must be available on PATH.

    Parameters
    ----------
    filepath_mol2 : path-like
        Path to the input MOL2 file.
    filepath_lt : path-like, optional
        Path to the output .lt file. If None, same stem as filepath_mol2 with .lt suffix.
    forcefield : {"GAFF2"}, default "GAFF2"
        Force field name. Used to select the force field file from moltemplate.
    name : str, default "UNL"
        Residue/molecule name written in the .lt file.

    Raises
    ------
    FileNotFoundError
        If filepath_mol2 does not exist or mol22lt.py is not found.
    subprocess.CalledProcessError
        If mol22lt.py exits with a non-zero status.
    """
    filepath_mol2 = Path(filepath_mol2)
    if filepath_lt is None:
        filepath_lt = filepath_mol2.with_suffix(".lt")
    else:
        filepath_lt = Path(filepath_lt)

    moltemplate_module = importlib.import_module("moltemplate")
    assert moltemplate_module.__file__ is not None
    filepath_forcefield = (
        Path(moltemplate_module.__file__).parent / "force_fields" / "gaff2.lt"
    )

    list_commands = [
        "mol22lt.py",
        "-i",
        str(filepath_mol2),
        "-o",
        str(filepath_lt),
        "-name",
        name,
        "--ff",
        forcefield,
        "--ff-file",
        str(filepath_forcefield),
    ]
    run_executable(list_commands)


def parse_mol_spec(mol_spec: MolSpec) -> tuple[str, Path, int]:
    """Extract (mol_name, filepath_lt, mol_count) from a MoleculeSpec or tuple.

    Parameters
    ----------
    mol_spec : tuple of (str, path-like, int)
        MoleculeSpec: (mol_name, filepath_lt, mol_count).

    Returns
    -------
    tuple of (str, Path, int)
        (mol_name, filepath_lt as Path, mol_count).

    Raises
    ------
    ValueError
        If mol_spec does not have the expected structure.
    """
    if (
        len(mol_spec) == 3
        and isinstance(mol_spec[0], str)
        and isinstance(mol_spec[1], (str, os.PathLike))
        and isinstance(mol_spec[2], int)
    ):
        mol_name, filepath_lt, mol_count = mol_spec
        filepath_lt = Path(filepath_lt).resolve()
        return mol_name, filepath_lt, mol_count
    raise ValueError(f"Invalid mol_spec: {mol_spec!r}")


def write_system_lt(
    mol_specs: Sequence[MolSpec],
    filepath_system_lt: Union[str, os.PathLike] = "./system.lt",
    cell_bounds: Optional[CellBounds] = None,
) -> None:
    """Write a Moltemplate system .lt file that imports molecule .lt files.

    Parameters
    ----------
    mol_specs : sequence of MolSpec
        Each element is (mol_name, filepath_lt, mol_count).
    filepath_system_lt : path-like, default "./system.lt"
        Output path for the system .lt file.
    cell_bounds : tuple of 3 (lo, hi) pairs, optional
        If given, writes a "Data Boundary" block with xlo xhi, ylo yhi, zlo zhi.
    """
    filepath_system_lt = Path(filepath_system_lt).resolve()

    import_statements: list[str] = []
    mol_defs: list[str] = []
    for mol_spec in mol_specs:
        mol_name, filepath_lt, mol_count = parse_mol_spec(mol_spec)
        import_statements.append('import "{}"'.format(filepath_lt))
        mol_defs.append(
            "{mol_name}_instance = new {mol_name} [{mol_count}]".format(
                mol_name=mol_name, mol_count=mol_count
            )
        )

    list_boundary: list[str] = []
    if cell_bounds:
        assert len(cell_bounds) == 3
        list_boundary.append('write_once("Data Boundary") {')
        for axis, (lo, hi) in zip(("x", "y", "z"), cell_bounds):
            list_boundary.append(f"  {lo: 7.4f} {hi: 7.4f} {axis}lo {axis}hi")
        list_boundary.append("}")

    filepath_system_lt.write_text(
        "\n".join(
            import_statements + [""] + mol_defs + [""] + list_boundary + [""]
        )
    )


def write_lammps_input(
    mol_and_num_chains: Sequence[tuple[Chem.Mol, int]],
    charges: Optional[Sequence[float]] = None,
    density: float = 0.3,
    seed: Optional[int] = None,
    work_in_cwd: bool = False,
) -> None:
    """Generate LAMMPS input files from molecules via Moltemplate.

    Writes MOL2 and .lt files for each molecule, builds an amorphous cell (or uses
    a single molecule), writes system.pdb and system.lt, then runs moltemplate.sh
    to produce LAMMPS data and input files.

    Parameters
    ----------
    mol_and_num_chains : sequence of (Mol, int)
        Each element is (RDKit Mol, number of chains to place in the system).
    charges : sequence of float, optional
        Per-atom charges for MOL2. If None, charges are assigned by the engine.
    density : float, default 0.3
        Target density (g/cm³) for amorphous cell and box size.
    seed : int, optional
        Random seed for amorphous cell generation.
    work_in_cwd : bool, default False
        If True, run MolToMol2File (antechamber) and write all intermediate files
        in the current directory. If False, intermediate files are written to a
        temporary directory; LAMMPS output files (system.data, system.in.*) are
        copied to the current directory.

    Raises
    ------
    RuntimeError
        If moltemplate.sh exits with a non-zero status.
    """
    cwd = Path.cwd().resolve()
    with work_directory(work_in_cwd) as work_dir:
        filepath_system_pdb = work_dir / "system.pdb"
        filepath_system_lt = work_dir / "system.lt"

        list_mol_specs: list[tuple[str, Path, int]] = []
        for idx, (mol, num_chains) in enumerate(mol_and_num_chains):
            name = f"MOL{idx}"
            filepath_mol2 = work_dir / f"{name}.mol2"

            MolToMol2File(
                mol,
                filepath_mol2,
                charges=charges,
                atom_type="gaff2",
                name=name,
                engine="antechamber",
                work_in_cwd=work_in_cwd,
            )
            mol22lt(filepath_mol2, forcefield="GAFF2", name=name)
            list_mol_specs.append(
                (name, filepath_mol2.with_suffix(".lt"), num_chains)
            )

        if len(mol_and_num_chains) > 1 or mol_and_num_chains[0][1] > 1:
            mol_system = generate_amorphous_cell(
                mol_and_num_chains,
                density=density,
                nloop=50,
                maxit=20,
                seed=seed,
            )
        else:
            mol_system = mol_and_num_chains[0][0]

        Chem.MolToPDBFile(mol_system, str(filepath_system_pdb))

        box_length = calculate_box_length(
            sum(atom.GetMass() for atom in mol_system.GetAtoms()),
            density=density,
        )
        box_length_half = box_length / 2

        write_system_lt(
            list_mol_specs,
            filepath_system_lt,
            cell_bounds=((-box_length_half, box_length_half),) * 3,
        )

        run_executable(
            [
                "moltemplate.sh",
                "-atomstyle",
                "full",
                "-pdb",
                str(filepath_system_pdb),
                str(filepath_system_lt),
            ],
            cwd=work_dir,
        )

        if not work_in_cwd:
            for filepath in work_dir.iterdir():
                if filepath.is_file() and (
                    any(
                        suffix in {".in", ".data"}
                        for suffix in filepath.suffixes
                    )
                ):
                    shutil.copy2(filepath, cwd / filepath.name)
