import io
import os
import re
from pathlib import Path
from typing import Literal, Union, overload

import networkx as nx
import numpy as np
import pandas as pd

from lammps_utils.constants import (
    COLS_ATOMS_LAMMPS_DATA_DTYPE,
    COLS_BONDS_LAMMPS_DATA_DTYPE,
    COLS_XYZ,
    MAP_ELEMENT_MASSES,
)
from lammps_utils.graph.pbc._pbc import (
    unwrap_positions_under_pbc,
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
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
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
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
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
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
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
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
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
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
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
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
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


def get_bond_dataframe(
    filepath_data_or_buffer: Union[
        os.PathLike, str, io.TextIOBase, io.BufferedIOBase
    ],
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


def get_atom_dataframe(
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
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

    return _df_atoms


def unwrap_df_positions_under_pbc(
    df_atoms: pd.DataFrame,
    df_bonds: pd.DataFrame,
    cell_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ],
) -> pd.DataFrame:
    """
    Adjust atomic coordinates to make the molecule whole under periodic boundary conditions (PBC).
    This function shifts atoms so that bonded atoms appear spatially close, avoiding discontinuities across cell edges.

    Parameters
    ----------
    df_atoms : pd.DataFrame
        DataFrame containing atomic coordinates. Must include columns "x", "y", and "z".
    df_bonds : pd.DataFrame
        DataFrame defining atomic bonds, with columns "atom1" and "atom2" containing atom indices.
    cell_bounds : tuple of tuple of float
        Bounds of the periodic cell along each axis. Format: ((xmin, xmax), (ymin, ymax), (zmin, zmax)).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with adjusted atomic coordinates that are spatially continuous across the cell.
    """

    # Ensure all coordinate columns have the same dtype
    st_dtypes = set(df_atoms.dtypes[COLS_XYZ])
    assert len(st_dtypes) == 1, (
        "Columns x, y, and z must have the same data type"
    )
    dtype = st_dtypes.pop()

    # Compute cell size in each direction
    cell_size = np.array(
        [bound[1] - bound[0] for bound in cell_bounds],
        dtype=dtype,
    )

    edges = (
        df_bonds[["atom1", "atom2"]]
        .apply(lambda col: df_atoms.index.get_indexer_for(col), axis=0)
        .values
    )

    # Build bond graph
    graph = nx.Graph()
    graph.add_edges_from(edges)

    # Prepare a new DataFrame for adjusted coordinates
    df_atoms_new = df_atoms.copy()

    df_atoms_new.loc[:, COLS_XYZ] = unwrap_positions_under_pbc(
        graph, df_atoms.loc[:, COLS_XYZ].values, cell_size=cell_size
    )

    return df_atoms_new
