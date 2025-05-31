import io
import math
import os
import re
from pathlib import Path
from typing import Literal, Optional, Union, overload

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from lammps_utils.constants import (
    COLS_ATOMS_LAMMPS_DATA_DTYPE,
    COLS_BONDS_LAMMPS_DATA_DTYPE,
    MAP_ELEMENT_MASSES,
)

PATTERN_N_ATOMS = r"\s*(\d+)\s+atoms\s*\n"
PATTERN_N_ATOM_TYPES = r"\s*(\d+)\s+atom types\s*\n"
PATTERN_N_BONDS = r"\s*(\d+)\s*bonds\s*\n"


@overload
def _read_file_or_buffer(
    filepath_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
    as_bytes: Literal[False] = False,
) -> str: ...


@overload
def _read_file_or_buffer(
    filepath_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
    as_bytes: Literal[True] = True,
) -> bytes: ...


def _read_file_or_buffer(
    filepath_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
    as_bytes: bool = False,
) -> Union[str, bytes]:
    """
    Read a LAMMPS data file or a file-like object and return its content as a string.
    This function handles both file paths and file-like objects (e.g., StringIO).
    If a file path is provided, it checks if the file exists and raises a FileNotFoundError

    Parameters
    ----------
    filepath_or_buffer : Union[str, os.PathLike, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    str
        The content of the file or file-like object as a string.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    TypeError
        If the input is neither a string, os.PathLike, nor a file-like object.
    """
    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        filepath_or_buffer = Path(filepath_or_buffer)
        if not filepath_or_buffer.is_file():
            raise FileNotFoundError(
                f"File {filepath_or_buffer} does not exist."
            )

        mode = "rb" if as_bytes else "r"
        with open(filepath_or_buffer, mode=mode) as f:
            return f.read()
    elif isinstance(filepath_or_buffer, io.TextIOBase):
        _out_txt = filepath_or_buffer.read()
        return _out_txt.encode() if as_bytes else _out_txt
    elif isinstance(filepath_or_buffer, io.BufferedIOBase):
        _out_bytes = filepath_or_buffer.read()
        return _out_bytes if as_bytes else _out_bytes.decode()
    else:
        raise TypeError(
            f"Expected str, os.PathLike or io.TextIOBase, got {type(filepath_or_buffer)}"
        )


def get_n_atoms(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
) -> int:
    """
    Get the number of atoms from a LAMMPS data file or a file-like object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[str, os.PathLike, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    int
        The number of atoms in the LAMMPS data file.

    Raises
    ------
    ValueError
        If the number of atoms cannot be found in the file.
    """
    content = _read_file_or_buffer(filepath_data_or_buffer)
    if _result_n_atoms := re.search(PATTERN_N_ATOMS, content):
        return int(_result_n_atoms.group(1))
    else:
        raise ValueError(
            f"Could not find number of atoms in {filepath_data_or_buffer}. "
            f"Make sure the file is a valid LAMMPS data file."
        )


def get_n_atom_types(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
) -> int:
    """
    Get the number of atom types from a LAMMPS data file or a file-like object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[str, os.PathLike, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    int
        The number of atom types in the LAMMPS data file.

    Raises
    ------
    ValueError
        If the number of atom types cannot be found in the file.
    """
    content = _read_file_or_buffer(filepath_data_or_buffer)
    if _result_n_atom_types := re.search(PATTERN_N_ATOM_TYPES, content):
        return int(_result_n_atom_types.group(1))
    else:
        raise ValueError(
            f"Could not find number of atom types in {filepath_data_or_buffer}. "
            f"Make sure the file is a valid LAMMPS data file."
        )


def get_n_bonds(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
) -> int:
    """
    Get the number of bonds from a LAMMPS data file or a file-like object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[str, os.PathLike, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    int
        The number of bonds in the LAMMPS data file.

    Raises
    ------
    ValueError
        If the number of bonds cannot be found in the file.
    """
    content = _read_file_or_buffer(filepath_data_or_buffer)
    if _result_n_bonds := re.search(PATTERN_N_BONDS, content):
        return int(_result_n_bonds.group(1))
    else:
        raise ValueError(
            f"Could not find number of bonds in {filepath_data_or_buffer}. "
            f"Make sure the file is a valid LAMMPS data file."
        )


def get_atom_type_masses(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
) -> dict[int, float]:
    """
    Get the masses of atom types from a LAMMPS data file or a file-like object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[str, os.PathLike, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    dict[int, float]
        A dictionary mapping atom type IDs to their masses.

    Raises
    ------
    ValueError
        If the atom type masses cannot be found in the file.
    """
    content = _read_file_or_buffer(filepath_data_or_buffer)
    n_atom_types = get_n_atom_types(io.StringIO(content))
    if _result_masses := re.search(
        r"\s*Masses.*\n\s*" + r"(\d+)\s+([\d\.]+).*\n\s*" * n_atom_types,
        content,
    ):
        return {
            int(_result_masses.group(2 * _i - 1)): float(
                _result_masses.group(2 * _i)
            )
            for _i in range(1, n_atom_types + 1)
        }
    else:
        raise ValueError(
            f"Could not find atom type masses in {filepath_data_or_buffer}. "
            f"Make sure the file is a valid LAMMPS data file."
        )


def get_atom_type_symbols(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
) -> dict[int, str]:
    """
    Get the symbols of atom types from a LAMMPS data file or a file-like object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[str, os.PathLike, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    dict[int, str]
        A dictionary mapping atom type IDs to their symbols.
    """
    atom_type_masses = get_atom_type_masses(filepath_data_or_buffer)
    sr_element_masses = pd.Series(MAP_ELEMENT_MASSES)
    return {
        _atom_id: sr_element_masses.index.tolist()[
            (sr_element_masses - _mass).abs().argmin()
        ]
        for _atom_id, _mass in atom_type_masses.items()
    }


def get_cell_bounds(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Get the cell limits from a LAMMPS data file or a file-like object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[str, os.PathLike, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        A tuple containing the cell limits for x, y, and z axes.
    """
    content = _read_file_or_buffer(filepath_data_or_buffer)

    _list_cell_bounds: list[tuple[float, float]] = []
    for x in ("x", "y", "z"):
        if _result_cell_bounds := re.search(
            rf"\s*([-+\d\.e]+)\s+([-+\d\.e]+)\s+{x}lo\s+{x}hi", content
        ):
            _list_cell_bounds.append(
                (
                    float(_result_cell_bounds.group(1)),
                    float(_result_cell_bounds.group(2)),
                )
            )
        else:
            raise ValueError(
                f"Could not find {x} cell bounds in {filepath_data_or_buffer}. "
                f"Make sure the file is a valid LAMMPS data file."
            )
    cell_bounds = tuple(_list_cell_bounds)
    assert len(cell_bounds) == 3
    return cell_bounds


def _get_bond_dataframe(
    filepath_data_or_buffer: Union[os.PathLike, str, io.TextIOBase],
) -> pd.DataFrame:
    """
    Get the bond DataFrame from a LAMMPS data file or a file-like object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[os.PathLike, str, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing bond data with columns for id, type, atom1, and atom2.

    Raises
    ------
    ValueError
        If the bond data cannot be found in the file.
    """
    content = _read_file_or_buffer(filepath_data_or_buffer)

    n_bonds = get_n_bonds(io.StringIO(content))
    if _result_bonds := re.search(
        r"\s*Bonds.*?\n(" + r"\s*\d+\s+\d+\s+\d+\s+\d+.*\n" * n_bonds + r")",
        content,
    ):
        return (
            (
                pd.read_table(
                    io.StringIO(_result_bonds.group(1)),
                    sep="\\s+",
                    header=None,
                )
                .rename(
                    columns=dict(
                        enumerate(COLS_BONDS_LAMMPS_DATA_DTYPE.keys())
                    )
                )
                .astype(COLS_BONDS_LAMMPS_DATA_DTYPE)
            )
            .set_index("id")
            .sort_index()
        )
    else:
        raise ValueError("Could not find bond data in the file.")


def _make_molecule_whole(
    df_atoms: pd.DataFrame, cell_size: ArrayLike
) -> pd.DataFrame:
    df_atoms_new = df_atoms.copy()
    cell_size = np.asarray(cell_size)
    XYZ_COLS = ["x", "y", "z"]

    for mol_id, group in df_atoms.groupby("mol"):
        idx = group.index
        coords = group[XYZ_COLS].to_numpy()

        # 相対変位を計算（基準はひとつ前の原子）
        deltas = np.diff(coords, axis=0)
        deltas -= cell_size * np.round(deltas / cell_size)  # 最小画像法補正

        # 絶対座標に再構成（基準はcoords[0]）
        new_coords = np.vstack(
            [coords[0], coords[0] + np.cumsum(deltas, axis=0)]
        )

        df_atoms_new.loc[idx, XYZ_COLS] = new_coords

    return df_atoms_new


def _get_atom_dataframe(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
    make_molecule_whole: bool = False,
) -> pd.DataFrame:
    """
    Get the atom DataFrame from a LAMMPS data file or a file-like object.

    Parameters
    ----------
    filepath_data_or_buffer : Union[str, os.PathLike, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing atom data with columns for id, mol, type, charge,
        x, y, z, and symbol.

    Raises
    ------
    ValueError
        If the atom data cannot be found in the file.
    """
    content = _read_file_or_buffer(filepath_data_or_buffer)

    n_atoms = get_n_atoms(io.StringIO(content))
    atom_type_symbols = get_atom_type_symbols(io.StringIO(content))

    if _result_atoms := re.search(
        r"\s*Atoms.*?\n("
        + r"\s*\d+\s+\d+\s+\d+\s+[-+\d\.e]+\s+[-+\d\.e]+\s+[-+\d\.e]+\s+[-+\d\.e]+.*\n"
        * n_atoms
        + r")",
        content,
    ):
        _df_atoms = (
            pd.read_table(
                io.StringIO(_result_atoms.group(1)), header=None, sep="\\s+"
            )
            .iloc[:, : len(COLS_ATOMS_LAMMPS_DATA_DTYPE)]
            .rename(
                columns=dict(enumerate(COLS_ATOMS_LAMMPS_DATA_DTYPE.keys()))
            )
            .astype(COLS_ATOMS_LAMMPS_DATA_DTYPE)
        )
    else:
        raise ValueError(
            f"Could not find atom data in {filepath_data_or_buffer}. "
            f"Make sure the file is a valid LAMMPS data file."
        )

    assert _df_atoms["type"].max() == get_n_atom_types(io.StringIO(content))
    _df_atoms["symbol"] = _df_atoms["type"].replace(atom_type_symbols)

    _df_atoms.set_index("id", inplace=True)
    _df_atoms.sort_index(inplace=True)

    if make_molecule_whole:
        _df_atoms = _make_molecule_whole(
            _df_atoms,
            cell_size=tuple(
                map(
                    lambda x: x[1] - x[0],
                    get_cell_bounds(io.StringIO(content)),
                )
            ),
        )

    return _df_atoms


@overload
def load_data(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
    make_molecule_whole: bool = False,
    return_bond_info: Literal[False] = False,
    return_cell_bounds: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def load_data(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
    make_molecule_whole: bool = False,
    return_bond_info: Literal[False] = False,
    return_cell_bounds: Literal[True] = True,
) -> tuple[
    pd.DataFrame,
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
]: ...
@overload
def load_data(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
    make_molecule_whole: bool = False,
    return_bond_info: Literal[True] = True,
    return_cell_bounds: Literal[False] = False,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


@overload
def load_data(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
    make_molecule_whole: bool = False,
    return_bond_info: Literal[True] = True,
    return_cell_bounds: Literal[True] = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
]: ...


def load_data(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
    make_molecule_whole: bool = False,
    return_bond_info: bool = False,
    return_cell_bounds: bool = False,
) -> Union[
    pd.DataFrame,
    tuple[pd.DataFrame, pd.DataFrame],
    tuple[
        pd.DataFrame,
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ],
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ],
]:
    """
    Load atom data from a LAMMPS data file or a file-like object into a DataFrame.

    Parameters
    ----------
    filepath_data_or_buffer : Union[str, os.PathLike, io.TextIOBase]
        The file path or file-like object to read.

    Returns
    -------
    Union[pd.DataFrame,tuple[pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    tuple[pd.DataFrame,pd.DataFrame,tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]]

        A DataFrame containing atom data, or a tuple of DataFrames and cell bounds
            if requested.
        If 'return_bond_info' is True, a tuple of two DataFrames (atom and bond data)
            is returned. If 'return_cell_bounds' is True, a tuple of three elements
            (atom DataFrame, bond DataFrame, and cell bounds) is returned.
        If both 'return_bond_info' and 'return_cell_bounds' are True, a tuple of
            three elements (atom DataFrame, bond DataFrame, and cell bounds) is returned.
        If both are False, only the atom DataFrame is returned.

    """
    content = _read_file_or_buffer(filepath_data_or_buffer)

    _list_out = [
        _get_atom_dataframe(
            io.StringIO(content), make_molecule_whole=make_molecule_whole
        )
    ]

    if return_bond_info:
        _list_out.append(_get_bond_dataframe(io.StringIO(content)))

    if return_cell_bounds:
        _list_out.append(get_cell_bounds(io.StringIO(content)))

    if len(_list_out) > 1:
        return tuple(_list_out)
    else:
        return _list_out[0]


@overload
def _parse_dump_timestep(
    filepath_dump_or_buffer: Union[
        os.PathLike, str, io.TextIOBase, io.BufferedIOBase
    ],
    return_cell_bounds: Literal[False] = False,
) -> tuple[int, pd.DataFrame]: ...


@overload
def _parse_dump_timestep(
    filepath_dump_or_buffer: Union[
        os.PathLike, str, io.TextIOBase, io.BufferedIOBase
    ],
    return_cell_bounds: Literal[True] = True,
) -> tuple[
    int,
    pd.DataFrame,
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
]: ...


@overload
def _parse_dump_timestep(
    filepath_dump_or_buffer: Union[
        os.PathLike, str, io.TextIOBase, io.BufferedIOBase
    ],
    return_cell_bounds: bool = False,
) -> Union[
    tuple[
        int,
        pd.DataFrame,
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ],
    tuple[int, pd.DataFrame],
]: ...


def _parse_dump_timestep(
    filepath_dump_or_buffer: Union[
        os.PathLike, str, io.TextIOBase, io.BufferedIOBase
    ],
    return_cell_bounds: bool = False,
) -> Union[
    tuple[
        int,
        pd.DataFrame,
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ],
    tuple[int, pd.DataFrame],
]:
    """
    Load and parse a LAMMPS dump file.

    This function reads a LAMMPS-style dump file and extracts:
    - The simulation timestep
    - A DataFrame of atom information
    - The simulation cell bounds (if requested)

    Parameters
    ----------
    filepath_dump_or_buffer : Union[os.PathLike, str, io.TextIOBase]
        Path to the dump file or a file-like buffer containing the dump content.
    return_cell_bounds : bool, optional
        Whether to return the simulation cell bounds. Defaults to False.

    Returns
    -------
    tuple
        If return_cell_bounds is False:
            (timestep, atom_dataframe)
        If return_cell_bounds is True:
            (timestep, atom_dataframe, cell_bounds)

        - timestep : int
            The simulation timestep extracted from the dump file.
        - atom_dataframe : pandas.DataFrame
            A DataFrame containing atom information indexed by atom ID.
        - cell_bounds : tuple of 3 tuples
            The simulation cell bounds in the format ((xlo, xhi), (ylo, yhi), (zlo, zhi)).

    Raises
    ------
    ValueError
        If any of the required sections (TIMESTEP, NUMBER OF ATOMS, BOX BOUNDS, ATOMS)
        are missing or malformed in the dump file.
    """
    content = _read_file_or_buffer(filepath_dump_or_buffer, as_bytes=True)

    if _match_timestep := re.search(rb"ITEM:\s+TIMESTEP\s+(\d+)", content):
        timestep = int(_match_timestep.group(1))
    else:
        raise ValueError("Failed to find TIMESTEP in the dump file.")

    if _match_n_atoms := re.search(
        rb"ITEM:\s+NUMBER OF ATOMS\s+(\d+)", content
    ):
        n_atoms = int(_match_n_atoms.group(1))
    else:
        raise ValueError("Failed to find NUMBER OF ATOMS in the dump file.")

    if _match_cell_bound := re.search(
        rb"ITEM:\s+BOX BOUNDS\s+.*\s*"
        + rb"([+-e\.\d]+)\s+([+-e\.\d]+)\s+" * 3,
        content,
    ):
        _list_cell_bounds: list[tuple[float, float]] = []
        for _idx in range(3):
            _list_cell_bounds.append(
                (
                    float(_match_cell_bound.group(2 * _idx + 1)),
                    float(_match_cell_bound.group(2 * _idx + 2)),
                )
            )
        cell_bounds = tuple(_list_cell_bounds)
        assert len(cell_bounds) == 3
    else:
        raise ValueError("Failed to find BOX BOUNDS in the dump file.")

    if _match_atoms := re.search(
        rb"ITEM: ATOMS (id type mol.*\n" + rb".+\n" * n_atoms + rb")", content
    ):
        df = pd.read_table(
            io.BytesIO(_match_atoms.group(1)),
            index_col=0,
            sep="\\s+",
        )
    else:
        raise ValueError("Failed to find ATOMS section in the dump file.")

    if return_cell_bounds:
        return (timestep, df, cell_bounds)
    else:
        return (timestep, df)


OVERWRAP = 50


def _find_timestep_offsets(
    filepath_dump: Union[os.PathLike, str],
    index: int,
    buffer_size: int = 10 * 1024 * 1024,
):
    filepath_dump = Path(filepath_dump)

    start = index * (buffer_size - OVERWRAP)
    with open(filepath_dump, mode="rb") as f:
        f.seek(start)
        return tuple(
            _match_timestep.start(0) + start
            for _match_timestep in re.finditer(
                rb"ITEM:\s+TIMESTEP\s+(\d+)", f.read(buffer_size)
            )
        )


def _load_timestep_chunk(
    filepath_dump,
    index_step: int,
    offsets: tuple[int, ...],
    return_cell_bounds: bool = False,
):
    with open(filepath_dump, mode="rb") as f:
        f.seek(offsets[index_step])
        return _parse_dump_timestep(
            io.BytesIO(
                f.read(
                    offsets[index_step + 1] - offsets[index_step]
                    if index_step < len(offsets) - 1
                    else None
                )
            ),
            return_cell_bounds=return_cell_bounds,
        )


def load_dump(
    filepath_dump: Union[os.PathLike, str],
    buffer_size: int = 10 * 1024 * 1024,
    return_cell_bounds: bool = False,
    n_jobs: Optional[int] = None,
):
    filepath_dump = Path(filepath_dump)
    if not filepath_dump.is_file():
        raise FileNotFoundError(filepath_dump)

    n = math.ceil((filepath_dump.stat().st_size + OVERWRAP) / buffer_size)

    offsets: tuple[int, ...] = tuple(
        sorted(
            set(
                sum(
                    Parallel(n_jobs=n_jobs)(
                        delayed(_find_timestep_offsets)(
                            filepath_dump, index, buffer_size=buffer_size
                        )
                        for index in range(n)
                    ),
                    start=tuple(),
                )
            )
        )
    )

    return sum(
        Parallel(n_jobs=n_jobs)(
            delayed(_load_timestep_chunk)(
                filepath_dump, index_step, offsets, return_cell_bounds
            )
            for index_step in range(len(offsets))
        ),
        start=tuple(),
    )
