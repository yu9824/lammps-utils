import io
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Union, overload

import pandas as pd
from joblib import Parallel, delayed

from lammps_utils.io.parsing._parsing import (
    _read_file_or_buffer,
    get_atom_dataframe,
    get_bond_dataframe,
    get_cell_bounds,
)


@overload
def load_data(
    filepath_data_or_buffer: Union[str, os.PathLike, io.TextIOBase],
    make_molecule_whole: bool = False,
    return_bond_info: Literal[False] = False,
    return_cell_bounds: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def load_data(
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
    make_molecule_whole: bool = False,
    return_bond_info: Literal[False] = False,
    return_cell_bounds: Literal[True] = True,
) -> tuple[
    pd.DataFrame,
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
]: ...
@overload
def load_data(
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
    make_molecule_whole: bool = False,
    return_bond_info: Literal[True] = True,
    return_cell_bounds: Literal[False] = False,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


@overload
def load_data(
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
    make_molecule_whole: bool = False,
    return_bond_info: Literal[True] = True,
    return_cell_bounds: Literal[True] = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
]: ...


def load_data(
    filepath_data_or_buffer: Union[
        str, os.PathLike, io.TextIOBase, io.BufferedIOBase
    ],
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
    Load atom (and optionally bond and cell) data from a LAMMPS data file into a DataFrame.

    This function supports file paths and file-like objects and optionally reconstructs molecules
    by unwrapping coordinates under periodic boundary conditions.

    Parameters
    ----------
    filepath_data_or_buffer : str or os.PathLike or io.TextIOBase or io.BufferedIOBase
        The path to a LAMMPS data file, or a file-like object containing the data.

    make_molecule_whole : bool, default=False
        If True, unwraps atomic coordinates using bond connectivity and periodic cell bounds
        so that molecules are made whole (not split across periodic boundaries).

    return_bond_info : bool, default=False
        If True, returns an additional DataFrame containing bond information.

    return_cell_bounds : bool, default=False
        If True, returns the simulation cell bounds as a tuple of 3 (min, max) pairs for x, y, z.

    Returns
    -------
    Union[pd.DataFrame,
    tuple[pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    tuple[pd.DataFrame, pd.DataFrame, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]
    ]

        The atom DataFrame is always returned. Depending on the flags:

        - If `return_bond_info` is True, bond DataFrame is included.
        - If `return_cell_bounds` is True, simulation box bounds are included.
        - If both flags are True, all three values are returned as a tuple:
          (atom DataFrame, bond DataFrame, cell bounds).

    Raises
    ------
    AssertionError
        If required components (like conformers) are missing when `make_molecule_whole` is True.

    Notes
    -----
    This function assumes the input file is in LAMMPS data format with sections for atoms,
    bonds, and box bounds. If `make_molecule_whole` is enabled, bond and box information
    are automatically parsed regardless of the other flags.
    """

    content = _read_file_or_buffer(filepath_data_or_buffer)

    _df_atoms = get_atom_dataframe(io.StringIO(content))

    if return_bond_info or make_molecule_whole:
        _df_bonds = get_bond_dataframe(io.StringIO(content))

    if return_cell_bounds or make_molecule_whole:
        _cell_bounds = get_cell_bounds(io.StringIO(content))

    if make_molecule_whole:
        from lammps_utils.io.dataframe._pbc import (
            unwrap_df_positions_under_pbc,
        )

        _df_atoms = unwrap_df_positions_under_pbc(
            df_atoms=_df_atoms, df_bonds=_df_bonds, cell_bounds=_cell_bounds
        )

    _list_out = [_df_atoms]
    if return_bond_info:
        _list_out.append(_df_bonds)

    if return_cell_bounds:
        _list_out.append(_cell_bounds)

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
        rb"ITEM: ATOMS (id .*\n" + rb".+\n" * n_atoms + rb")", content
    ):
        df = pd.read_table(
            io.BytesIO(_match_atoms.group(1)),
            index_col=0,
            sep="\\s+",
        )
    else:
        raise ValueError("Failed to find ATOMS section in the dump file.")

    for idx_axis, axis in enumerate(("x", "y", "z")):
        if axis in df.columns:
            continue

        col = f"{axis}s"
        if col in df.columns:
            df.loc[:, axis] = (
                df.loc[:, col]
                * (cell_bounds[idx_axis][1] - cell_bounds[idx_axis][0])
                + cell_bounds[idx_axis][0]
            )

    if return_cell_bounds:
        return (timestep, df, cell_bounds)
    else:
        return (timestep, df)


OVERWRAP = 50


def _find_timestep_offsets(
    filepath_dump: Union[os.PathLike, str],
    index: int,
    buffer_size: int = 10 * 1024 * 1024,
) -> tuple[int, ...]:
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
) -> Union[
    tuple[
        int,
        pd.DataFrame,
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ],
    tuple[int, pd.DataFrame],
]:
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


def _filter_timesteps(
    timestep_records: Union[
        tuple[tuple[int, pd.DataFrame], ...],
        tuple[
            tuple[
                int,
                pd.DataFrame,
                tuple[
                    tuple[float, float],
                    tuple[float, float],
                    tuple[float, float],
                ],
            ],
            ...,
        ],
    ],
    select: Optional[Union[int, slice, Sequence[int]]],
    select_by: Literal["timestep", "index"],
) -> Union[
    tuple[tuple[int, pd.DataFrame], ...],
    tuple[
        tuple[
            int,
            pd.DataFrame,
            tuple[
                tuple[float, float], tuple[float, float], tuple[float, float]
            ],
        ],
        ...,
    ],
]:
    """
    Filter timestep data based on select and select_by parameters.

    Parameters
    ----------
    timestep_records : tuple
        Tuple of (timestep, df_atoms) or (timestep, df_atoms, cell_bounds) tuples.
    select : Optional[Union[int, slice, Sequence[int]]]
        Selection criteria. If None, returns all frames.
    select_by : Literal["timestep", "index"]
        Whether to select by timestep value or index position.

    Returns
    -------
    tuple
        Filtered timestep data.
    """
    if select is None:
        return timestep_records

    if select_by == "timestep":
        # Create a mapping from timestep to index
        timestep_to_index = {
            frame: idx for idx, (frame, *_) in enumerate(timestep_records)
        }

        if isinstance(select, int):
            # Single timestep
            if select in timestep_to_index:
                idx = timestep_to_index[select]
                return (timestep_records[idx],)  # type: ignore[return-value]
            else:
                return timestep_records[:0]  # type: ignore[return-value]
        elif isinstance(select, slice):
            # Slice by timestep values - filter by timestep range
            frames = [frame for frame, *_ in timestep_records]
            if not frames:
                return timestep_records[:0]  # type: ignore[return-value]

            # Get slice bounds
            start = select.start if select.start is not None else min(frames)
            stop = select.stop if select.stop is not None else max(frames) + 1
            step = select.step if select.step is not None else 1

            # Filter by timestep range
            filtered = [
                (frame, idx)
                for idx, frame in enumerate(frames)
                if start <= frame < stop
            ]

            # Apply step
            if step > 0:
                filtered = filtered[::step]
            else:
                # For negative step, reverse the list and use positive step
                filtered = filtered[::-1][:: abs(step)]

            return tuple(timestep_records[idx] for _, idx in filtered)  # type: ignore[return-value]
        elif isinstance(select, Sequence):
            # Sequence of timestep_records
            return tuple(  # type: ignore[return-value]
                timestep_records[timestep_to_index[ts]]
                for ts in set(select)
                if ts in timestep_to_index
            )
    else:  # select_by == "index"
        if isinstance(select, int):
            # Single index
            if 0 <= select < len(timestep_records):
                return (timestep_records[select],)  # type: ignore[return-value]
            else:
                return timestep_records[:0]  # type: ignore[return-value]
        elif isinstance(select, slice):
            # Slice by index
            return timestep_records[select]  # type: ignore[return-value]
        elif isinstance(select, Sequence):
            # Sequence of indices
            return tuple(  # type: ignore[return-value]
                timestep_records[idx]
                for idx in set(select)
                if 0 <= idx < len(timestep_records)
            )

    return timestep_records


@overload
def load_dump(
    filepath_dump: Union[os.PathLike, str],
    select: Optional[Union[int, slice, Sequence[int]]] = None,
    select_by: Literal["timestep", "index"] = "timestep",
    buffer_size: int = 10 * 1024 * 1024,
    return_cell_bounds: Literal[False] = False,
    n_jobs: Optional[int] = None,
) -> tuple[tuple[int, pd.DataFrame], ...]: ...


@overload
def load_dump(
    filepath_dump: Union[os.PathLike, str],
    select: Optional[Union[int, slice, Sequence[int]]] = None,
    select_by: Literal["timestep", "index"] = "timestep",
    buffer_size: int = 10 * 1024 * 1024,
    return_cell_bounds: Literal[True] = True,
    n_jobs: Optional[int] = None,
) -> tuple[
    tuple[
        int,
        pd.DataFrame,
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ],
    ...,
]: ...


def load_dump(
    filepath_dump: Union[os.PathLike, str],
    select: Optional[Union[int, slice, Sequence[int]]] = None,
    select_by: Literal["timestep", "index"] = "timestep",
    buffer_size: int = 10 * 1024 * 1024,
    return_cell_bounds: bool = False,
    n_jobs: Optional[int] = None,
) -> Union[
    tuple[
        tuple[
            int,
            pd.DataFrame,
            tuple[
                tuple[float, float], tuple[float, float], tuple[float, float]
            ],
        ],
        ...,
    ],
    tuple[tuple[int, pd.DataFrame], ...],
]:
    """
    Load and parse a LAMMPS dump file into structured timestep data.

    Parameters
    ----------
    filepath_dump : Union[os.PathLike, str]
        Path to the LAMMPS dump file to be loaded.
    select : Optional[Union[int, slice, Sequence[int]]], optional
        Selection criteria for frames. If None, all frames are loaded.
        - If int: select a single frame
        - If slice: select a range of frames
        - If Sequence[int]: select specific frames
    select_by : Literal["timestep", "index"], optional
        Whether to select by timestep value ("timestep") or index position ("index").
        Default is "timestep".
    buffer_size : int, optional
        Size of the buffer to use when scanning the file for timestep offsets (in bytes).
        Larger values may improve performance on large files. Default is 10 MB.
    return_cell_bounds : bool, optional
        Whether to extract and return cell bounds for each timestep. If True, the output
        will include cell bounds in addition to timestep and atomic data. Default is False.
    n_jobs : Optional[int], optional
        Number of parallel jobs to run. If None, defaults to single-threaded operation.

    Returns
    -------
    Union[
    tuple[tuple[int, pd.DataFrame, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]], ...],
    tuple[tuple[int, pd.DataFrame], ...]
    ]
        A tuple of timestep data. Each element is either:
        - (timestep, DataFrame) if return_cell_bounds is False
        - (timestep, DataFrame, cell_bounds) if return_cell_bounds is True

        `timestep` is an integer, `DataFrame` contains atomic data for that step,
        and `cell_bounds` is a 3-tuple of (min, max) pairs for x, y, z.
    """
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

    timestep_records = tuple(
        Parallel(n_jobs=n_jobs)(
            delayed(_load_timestep_chunk)(
                filepath_dump, index_step, offsets, return_cell_bounds
            )
            for index_step in range(len(offsets))
        ),
    )

    # Filter timestep_records based on select and select_by
    return _filter_timesteps(timestep_records, select, select_by)
