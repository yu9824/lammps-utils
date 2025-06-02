import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem

from lammps_utils.graph._pbc import unwrap_molecule_under_pbc
from lammps_utils.rdkit._bond import get_bond_order


def unwrap_rdkit_mol_under_pbc(
    mol: Chem.rdchem.Mol,
    cell_size: ArrayLike,
    confId: int = -1,
    determine_bonds: bool = False,
) -> Chem.rdchem.Mol:
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
