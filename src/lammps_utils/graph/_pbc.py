import networkx as nx
import numpy as np
import pandas as pd

COLS_XYZ = ["x", "y", "z"]
"""Coordinate column labels

``["x", "y", "z"]``
"""


def unwrap_molecule_positions(
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

    # Build bond graph
    graph = nx.Graph()
    graph.add_edges_from(df_bonds[["atom1", "atom2"]].values)

    # Prepare a new DataFrame for adjusted coordinates
    df_atoms_new = df_atoms.copy()

    # Process each connected component (i.e., molecule or fragment)
    for component in nx.connected_components(graph):
        subgraph: nx.Graph = graph.subgraph(component)
        src = next(iter(subgraph.nodes), None)  # Root for BFS

        # Traverse the graph in breadth-first order to adjust positions
        bfs_tree: nx.DiGraph = nx.bfs_tree(subgraph, src)
        for idx_atom1, idx_atom2 in bfs_tree.edges:
            # Get reference coordinates (parent atom)
            ref = df_atoms_new.loc[idx_atom1, COLS_XYZ].values.astype(dtype)
            # Compute vector from reference to target atom
            vec = (
                df_atoms_new.loc[idx_atom2, COLS_XYZ].values.astype(dtype)
                - ref
            )
            # Apply periodic correction to vector
            delta = cell_size * np.round(vec / cell_size)
            corrected_vec = vec - delta
            # Update coordinates of the target atom
            df_atoms_new.loc[idx_atom2, COLS_XYZ] = ref + corrected_vec

    return df_atoms_new
