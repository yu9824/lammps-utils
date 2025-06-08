from typing import Optional

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem

from lammps_utils.graph._pbc import (
    unwrap_molecule_under_pbc,
    wrap_positions_to_cell,
)
from lammps_utils.rdkit._bond import get_bond_order


def unwrap_rdkit_mol_under_pbc(
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
    conditions (PBC), using the `unwrap_molecule_under_pbc` utility. If `determine_bonds` is True,
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
    positions_new = unwrap_molecule_under_pbc(
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


def wrap_mol_positions_to_cell(
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
        wrap_positions_to_cell(conf.GetPositions(), cell_bounds=cell_bounds)
    )
    return mol_new
