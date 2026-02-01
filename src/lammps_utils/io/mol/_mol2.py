import os
import re
import shutil
from collections.abc import Sequence
from typing import Literal, Optional, Union

from rdkit import Chem

from lammps_utils.helpers import check_encoding, run_executable, work_directory
from lammps_utils.logging import get_child_logger

logger = get_child_logger(__name__)

BIN_OBABEL = os.environ.get("BIN_OBABEL", "obabel")
BIN_ANTECHAMBER = os.environ.get("BIN_ANTECHAMBER", "antechamber")


def _mol_to_mol2_block_antechamber(
    mol: Chem.Mol,
    charges: Optional[Sequence[float]] = None,
    encoding: Optional[str] = None,
    atom_type: Literal["gaff", "gaff2", "sybyl", "bcc", "amber"] = "gaff2",
    name: Optional[str] = None,
    work_in_cwd: bool = False,
) -> str:
    """Convert an RDKit Mol object to a MOL2 block string using antechamber.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule to be converted.
    charges : Optional[Sequence[float]], optional
        If None (the default), no partial charges are added. If a sequence of floats is provided,
        these are used as the per-atom charges, and will be inserted into the MOL2 output.
    encoding : Optional[str], optional
        Character encoding to use for input and output. If None, system default encoding is used.
    atom_type : Literal[&quot;gaff&quot;, &quot;gaff2&quot;, &quot;sybyl&quot;, &quot;bcc&quot;, &quot;amber&quot;], optional
        The atom type to use for the MOL2 output.
        Supported atom types are 'gaff', 'gaff2', 'sybyl', 'bcc', and 'amber'.
    name : Optional[str], optional
        The name of the molecule. If None, the name of the molecule is not set.
    work_in_cwd : bool, optional
        If True, use the current working directory for intermediate files (input.pdb,
        output.mol2, etc.) instead of a temporary directory. Useful for debugging.
        Default is False.
    Returns
    -------
    str
        The MOL2 block string.

    Raises
    ------
    RuntimeError
        If the command fails.
    """
    if not shutil.which(BIN_ANTECHAMBER):
        raise FileNotFoundError(
            f"antechamber executable not found: {BIN_ANTECHAMBER}. "
            "Please install it using `conda install -c conda-forge ambertools` "
            "or set the `BIN_ANTECHAMBER` environment variable to the path of the antechamber executable."
        )

    if len(Chem.GetMolFrags(mol)) > 1:
        raise ValueError(
            "Multiple molecules are not supported for antechamber engine. "
            "Use obabel engine instead."
        )

    encoding = check_encoding(encoding)

    with work_directory(work_in_cwd) as work_dir:
        filepath_pdb = work_dir / "input.pdb"
        filepath_mol2 = work_dir / "output.mol2"

        Chem.MolToPDBFile(mol, str(filepath_pdb))
        list_commands = [
            BIN_ANTECHAMBER,
            "-i",
            str(filepath_pdb),
            "-fi",
            "pdb",
            "-o",
            str(filepath_mol2),
            "-fo",
            "mol2",
            "-at",
            atom_type,
        ]
        if name is not None:
            list_commands.extend(["-rn", name])
        if charges is not None:
            assert len(charges) == mol.GetNumAtoms(), (
                "charges must be a sequence of length equal to the number of atoms"
            )
            filepath_charges = work_dir / "charges.txt"
            filepath_charges.write_text("\n".join(map(str, charges)))
            list_commands.extend(["-c", "rc", "-cf", str(filepath_charges)])
        run_executable(list_commands, cwd=work_dir)
        return filepath_mol2.read_text(encoding=encoding)


def _mol_to_mol2_block_obabel(
    mol: Chem.Mol,
    charges: Optional[Sequence[float]] = None,
    encoding: Optional[str] = None,
) -> str:
    """
    Convert an RDKit Mol object to a MOL2 block string, optionally adding partial charges.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule to be converted.
    charges : Optional[Sequence[float]], default=None
        If None (the default), no partial charges are added. If a sequence of floats is provided,
        these are used as the per-atom charges, and will be inserted into the MOL2 output.
    encoding : Optional[str], default=None
        Character encoding to use for input and output. If None, system default encoding is used.

    Returns
    -------
    str
        MOL2 format block as a string, with the requested charges set.

    Notes
    -----
    - Relies on Open Babel (`obabel`) to perform the actual conversion to the MOL2 format.
    - Atom properties (including updated charges) are preserved in the output.
    - Custom charges will overwrite any charges written by Open Babel in the output.

    """
    if not shutil.which(BIN_OBABEL):
        raise FileNotFoundError(
            f"obabel executable not found: {BIN_OBABEL}. "
            "Please install it using `conda install -c conda-forge openbabel` "
            "or set the `BIN_OBABEL` environment variable to the path of the obabel executable."
        )

    REGEX_ATOM_LINE = re.compile(r"^(\s*\d+)(.+)( 0\.0+)$")
    # the space before 0.0 is required for the charge field to include the sign

    KEY_START = "@<TRIPOS>ATOM"
    KEY_END = "@<TRIPOS>BOND"

    replace_charges = False
    if charges is None:
        partial_charge_option = "none"
    elif charges == "gasteiger":
        partial_charge_option = "gasteiger"
    else:
        assert len(charges) == mol.GetNumAtoms(), (
            "The number of charges must be equal to the number of atoms"
        )
        partial_charge_option = "none"
        replace_charges = True

    encoding = check_encoding(encoding)

    list_commands = [
        BIN_OBABEL,
        "-ipdb",
        "--partialcharge",
        partial_charge_option,
        "-omol2",
    ]
    result_obabel = run_executable(list_commands)

    flag_read = False
    atom_index: int = 1  # 1-based index

    raw_mol2_block = result_obabel.stdout.decode(encoding)
    if replace_charges:
        # mypy cannot infer the type of charges when it is not None
        # so we need to assert it here
        assert charges is not None
        assert not isinstance(charges, str)

        lines: list[str] = []
        for line in raw_mol2_block.splitlines():
            if flag_read:
                if match_result := REGEX_ATOM_LINE.search(line):
                    assert int(match_result.group(1)) == atom_index
                    lines.append(
                        REGEX_ATOM_LINE.sub(
                            lambda match: "{}{}{:> 7.4f}".format(
                                match.group(1),
                                match.group(2),
                                charges[atom_index - 1],
                            ),
                            line,
                        )
                    )
                    atom_index += 1
                    continue
                elif line.startswith(KEY_END):
                    flag_read = False
            elif line.startswith(KEY_START):
                flag_read = True

            lines.append(line)
        return "\n".join(lines)
    else:
        return raw_mol2_block


def MolToMol2Block(
    mol: Chem.Mol,
    charges: Optional[Sequence[float]] = None,
    encoding: Optional[str] = None,
    atom_type: Literal["gaff", "gaff2", "sybyl", "bcc", "amber"] = "gaff2",
    name: Optional[str] = None,
    engine: Literal["antechamber", "obabel"] = "antechamber",
    work_in_cwd: bool = False,
) -> str:
    if (
        engine == "antechamber"
    ):  # not accept multiple molecules, but have mutch formats
        return _mol_to_mol2_block_antechamber(
            mol, charges, encoding, atom_type, name, work_in_cwd
        )
    elif (
        engine == "obabel"
    ):  # accept multiple molecules, but only support 'sybyl' format
        if atom_type != "sybyl":
            raise ValueError("atom_type must be 'sybyl' for obabel engine")

        if name is not None:
            logger.warning(
                "name is not supported for obabel engine. It will be ignored."
            )
        return _mol_to_mol2_block_obabel(mol, charges, encoding)
    else:
        raise ValueError("Invalid engine: {}".format(engine))


def MolToMol2File(
    mol: Chem.Mol,
    filepath: Union[os.PathLike, str],
    charges: Optional[Sequence[float]] = None,
    encoding: Optional[str] = None,
    atom_type: Literal["gaff", "gaff2", "sybyl", "bcc", "amber"] = "gaff2",
    name: Optional[str] = None,
    engine: Literal["antechamber", "obabel"] = "antechamber",
    work_in_cwd: bool = False,
) -> None:
    """
    Write an RDKit Mol object to a MOL2 file, optionally adding partial charges.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule to be converted.
    filepath : Union[os.PathLike, str]
        The path to the MOL2 file to write.
    charges : Optional[Sequence[float]], default=None
        If None (the default), no partial charges are added. If a sequence of floats is provided,
        these are used as the per-atom charges, and will be inserted into the MOL2 output.
    encoding : Optional[str], optional
        Character encoding to use for input and output. If None, system default encoding is used.
    atom_type : Literal["gaff", "gaff2", "sybyl", "bcc", "amber"], optional
        The atom type to use for the MOL2 output.
        Supported atom types are 'gaff', 'gaff2', 'sybyl', 'bcc', and 'amber'.
    name : Optional[str], optional
        The name of the molecule. If None, the name of the molecule is not set.
    engine : Literal["antechamber", "obabel"], optional
        The engine to use for the conversion.
        Supported engines are 'antechamber' and 'obabel'.
        Default is 'antechamber'.
    work_in_cwd : bool, optional
        If True (antechamber engine only), use the current working directory for
        intermediate files instead of a temporary directory. Useful for debugging.
        Default is False.
    """
    encoding = check_encoding(encoding)
    with open(filepath, mode="w", encoding=encoding) as f:
        f.write(
            MolToMol2Block(
                mol, charges, encoding, atom_type, name, engine, work_in_cwd
            )
        )
