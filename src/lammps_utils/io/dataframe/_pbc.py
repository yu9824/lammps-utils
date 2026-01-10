"""Module for handling periodic boundary conditions with pandas DataFrames."""

from typing import Union

import networkx as nx
import numpy as np
import pandas as pd

from lammps_utils.constants import COLS_XYZ
from lammps_utils.graph.pbc._pbc import (
    unwrap_positions_under_pbc,
    wrap_positions_into_cell,
)


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


def wrap_df_positions_to_cell(
    df_atoms: pd.DataFrame,
    cell_bounds: Union[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        np.ndarray,
    ],
) -> pd.DataFrame:
    """
    Wrap atomic positions in a DataFrame into the periodic simulation cell.

    This function modifies the input DataFrame by wrapping atomic positions
    so that all atoms are located within the given periodic cell boundaries.

    Parameters
    ----------
    df_atoms : pd.DataFrame
        A DataFrame containing atomic coordinates. Must include columns "x", "y", and "z".
    cell_bounds : Union[tuple, np.ndarray]
        The simulation cell bounds specified as a tuple of ((xlo, xhi), (ylo, yhi), (zlo, zhi)),
        or as a NumPy array of shape (3, 2).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with wrapped atomic coordinates within the cell.
    """
    df_atoms.loc[:, COLS_XYZ] = wrap_positions_into_cell(
        df_atoms.loc[:, COLS_XYZ], cell_bounds=cell_bounds
    )
    return df_atoms

