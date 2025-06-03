from collections.abc import Generator

import networkx as nx
from rdkit import Chem

from lammps_utils.graph._main_chain import _bfs_farthest_node, nodes_in_cycles


def find_main_chains(
    mol: Chem.rdchem.Mol,
) -> Generator[Chem.rdchem.Mol, None, None]:
    """
    Extracts and yields the longest linear (main chain) fragments from a molecule.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        An RDKit Mol object representing the molecular structure.

    Yields
    ------
    Chem.rdchem.Mol
        A fragment corresponding to the main chain (as a substructure) for each connected component.

    Raises
    ------
    TypeError
        If the input is not an RDKit Mol object.

    Notes
    -----
    This function identifies the longest acyclic paths (main chains) within each connected component
    of the input molecule.

    - Hydrogen atoms are removed prior to analysis.
    - An undirected graph is constructed from the molecule using the adjacency matrix.
    - Cyclic atoms are ignored when determining the main chain endpoints.
    - The longest path between two non-cyclic atoms is obtained via a breadth-first search,
      followed by a shortest path search in the graph.
    - The corresponding substructure (bond path) is returned as an RDKit Mol fragment.

    This method is useful for isolating backbone structures or linear scaffolds in a molecule.
    """

    if isinstance(mol, Chem.rdchem.Mol):
        mol = Chem.RemoveHs(mol)
        graph: nx.Graph = nx.from_numpy_array(
            Chem.GetAdjacencyMatrix(mol),
            # nodelist=range(1, mol.GetNumAtoms() + 1),
        )
    else:
        raise TypeError

    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        start, end = _bfs_farthest_node(
            subgraph,
            ignore_nodes=nodes_in_cycles(subgraph),
            return_length=False,
        )

        atom_indexes = tuple(nx.shortest_path(subgraph, start, end))
        path: list[int] = []
        for i_atom in range(len(atom_indexes) - 1):
            path.append(
                mol.GetBondBetweenAtoms(
                    atom_indexes[i_atom], atom_indexes[i_atom + 1]
                ).GetIdx()
            )
        yield Chem.PathToSubmol(mol, path)
