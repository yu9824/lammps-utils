from typing import Optional, Union

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem

from lammps_utils.chem.bond._bond import get_bond_order


def unwrap_positions_under_pbc(
    graph: nx.Graph, positions: np.ndarray, cell_size: ArrayLike
) -> np.ndarray:
    """
    Unwrap molecular coordinates under periodic boundary conditions (PBC).

    This function traverses the molecular graph and adjusts atomic positions so that
    bonded atoms are placed close together, eliminating jumps caused by PBC wrapping.
    It operates independently on each connected component of the graph.

    Parameters
    ----------
    graph : nx.Graph
        A molecular graph where nodes correspond to atoms and edges represent bonds.
        Each connected component is treated as an independent molecule or fragment.
    positions : np.ndarray
        A (N, 3) array of atomic coordinates, where N is the number of atoms.
    cell_size : ArrayLike
        A 1D array-like of length 3 specifying the dimensions of the periodic simulation box.

    Returns
    -------
    np.ndarray
        A (N, 3) NumPy array of unwrapped atomic coordinates. The coordinates are adjusted
        such that bonded atoms are positioned contiguously within the same image of the unit cell.

    Raises
    ------
    AssertionError
        If input dimensions are invalid or if the number of atoms in `graph` and `positions` do not match.

    Notes
    -----
    This method assumes that atoms are initially located within the same periodic image, and it
    corrects discontinuities across periodic boundaries by walking through the molecular graph
    using a breadth-first traversal.
    """

    assert positions.ndim == 2
    assert len(graph.nodes) == positions.shape[0]
    assert positions.shape[1] == 3

    cell_size = np.asarray(cell_size)
    assert cell_size.ndim == 1
    assert cell_size.shape[0] == 3

    positions_new = positions.copy()

    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        assert isinstance(subgraph, nx.Graph)
        src = next(iter(subgraph.nodes), None)  # Root for BFS

        # Traverse the graph in breadth-first order to adjust positions
        bfs_tree = nx.bfs_tree(subgraph, src)
        assert isinstance(bfs_tree, nx.DiGraph)
        for idx_atom1, idx_atom2 in bfs_tree.edges:
            # Get reference coordinates (parent atom)
            ref = positions_new[idx_atom1]
            # Compute vector from reference to target atom
            vec = positions_new[idx_atom2] - ref
            # Apply periodic correction to vector
            delta = cell_size * np.round(vec / cell_size)
            corrected_vec = vec - delta
            # Update coordinates of the target atom
            positions_new[idx_atom2] = ref + corrected_vec

    return positions_new


def wrap_positions_into_cell(
    positions: np.ndarray,
    cell_bounds: Union[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        np.ndarray,
    ],
) -> np.ndarray:
    """
    Wrap 3D positions into a periodic simulation cell.

    This function takes an array of 3D Cartesian coordinates and wraps each
    position into the simulation cell defined by the given bounds using
    periodic boundary conditions.

    Parameters
    ----------------
    positions : np.ndarray
        A NumPy array of shape (N, 3) representing the positions of N atoms.
    cell_bounds : tuple or np.ndarray
        The simulation cell bounds. Can be provided as a tuple of
        ((xlo, xhi), (ylo, yhi), (zlo, zhi)) or a NumPy array of shape (3, 2).

    Returns
    ----------------
    np.ndarray
        A NumPy array of shape (N, 3) containing the wrapped positions.
    """
    cell_bounds = np.asarray(cell_bounds)
    cell_min = cell_bounds[:, 0]
    # cell_min = np.array([b[0] for b in cell_bounds])  # shape (3,)
    cell_max = cell_bounds[:, 1]
    # cell_max = np.array([b[1] for b in cell_bounds])  # shape (3,)
    cell_range = cell_max - cell_min  # shape (3,)

    # broadcastingにより (N, 3) から直接演算
    return (positions - cell_min) % cell_range + cell_min


def unwrap_mol_under_pbc(
    mol: Chem.rdchem.Mol,
    cell_size: ArrayLike,
    confId: int = -1,
    determine_bonds: bool = False,
) -> Chem.rdchem.Mol:
    """
    Unwraps a periodic RDKit molecule so that bonded atoms are positioned close together in Cartesian space.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The RDKit molecule to be unwrapped. Must have at least one 3D conformer.
    cell_size : ArrayLike
        The size of the periodic simulation cell (a 3-element array-like object representing the box dimensions).
    confId : int, optional
        The conformer ID to use for coordinate manipulation. Defaults to -1 (the first conformer).
    determine_bonds : bool, optional
        If True, reassigns bond orders based on interatomic distances after unwrapping. Defaults to False.

    Returns
    -------
    Chem.rdchem.Mol
        A new RDKit molecule object with unwrapped coordinates and optionally updated bond orders.
        All hydrogen atoms are removed from the returned molecule.

    Raises
    ------
    AssertionError
        If the input molecule has no conformers or if the cell size is invalid.

    Notes
    -----
    This function converts the molecule to a graph to assist in unwrapping it under periodic boundary
    conditions (PBC), using the `unwrap_positions_under_pbc` utility. If `determine_bonds` is True,
    bond distances are recalculated post-unwrapping, and bond types are reassigned using the
    `get_bond_order` function. Hydrogens are removed from the returned molecule to simplify further processing.
    """

    assert mol.GetNumConformers() > 0
    rwmol = Chem.RWMol(mol)

    cell_size = np.asarray(cell_size)
    assert cell_size.shape[0] == 3
    assert cell_size.ndim == 1

    graph = nx.from_numpy_array(Chem.GetAdjacencyMatrix(rwmol))
    assert isinstance(graph, nx.Graph)
    conf = rwmol.GetConformer(confId)
    positions_new = unwrap_positions_under_pbc(
        graph, positions=conf.GetPositions(), cell_size=cell_size
    )
    conf.SetPositions(positions_new)

    if determine_bonds:
        for bond in rwmol.GetBonds():
            assert isinstance(bond, Chem.rdchem.Bond)
            distance = np.sqrt(
                np.sum(
                    np.square(
                        positions_new[bond.GetBeginAtomIdx()]
                        - positions_new[bond.GetEndAtomIdx()]
                    )
                )
            )
            bond.SetBondType(
                get_bond_order(
                    (
                        bond.GetBeginAtom().GetSymbol(),
                        bond.GetEndAtom().GetSymbol(),
                    ),
                    distance,
                )
            )

    return Chem.RemoveHs(
        rwmol.GetMol(),
        implicitOnly=True,
        updateExplicitCount=True,
        sanitize=True,
    )


def wrap_mol_into_cell(
    mol: Chem.rdchem.Mol,
    confId: int = -1,
    cell_bounds: Optional[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = None,
) -> Chem.rdchem.Mol:
    """
    Wrap atom positions of a conformer into the periodic simulation cell.

    This function returns a copy of the given RDKit molecule, with the specified
    conformer's atomic positions wrapped so that all atoms lie within the given
    simulation cell bounds. If `cell_bounds` is not provided, they will be inferred
    from the conformer's properties.

    Parameters
    ----------------
    mol : Chem.Mol
        An RDKit molecule containing at least one conformer.
    confId : int
        The conformer ID to wrap. Defaults to -1 (the last conformer).
    cell_bounds : tuple of 3 (lo, hi) tuples, optional
        The simulation cell bounds along x, y, and z axes. If not provided,
        they will be read from the conformer's properties: "xlo", "xhi", "ylo", etc.

    Returns
    ----------------
    Chem.Mol
        A copy of the input molecule with wrapped conformer coordinates.
    """
    mol_new = Chem.Mol(mol)
    conf = mol_new.GetConformer(confId)
    if cell_bounds is None:
        if not conf.HasProp("xlo"):
            raise ValueError

        _tup_tmp = tuple(
            (conf.GetDoubleProp(f"{axis}lo"), conf.GetDoubleProp(f"{axis}hi"))
            for axis in ("x", "y", "z")
        )
        assert len(_tup_tmp) == 3
        cell_bounds = _tup_tmp

    conf.SetPositions(
        wrap_positions_into_cell(conf.GetPositions(), cell_bounds=cell_bounds)
    )
    return mol_new
