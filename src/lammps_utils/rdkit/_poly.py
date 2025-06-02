from collections.abc import Generator
from typing import Union

import networkx as nx
from rdkit import Chem

from lammps_utils.graph._main_chain import _bfs_farthest_node, nodes_in_cycles


def find_main_chains(
    mol_or_graph: Union[Chem.rdchem.Mol, nx.Graph],
) -> Generator[tuple[int, ...], None, None]:
    """
    Generates main chain fragments from a molecular structure or graph.

    Parameters
    ----------
    mol_or_graph : Union[Chem.rdchem.Mol, nx.Graph]
        A molecular structure (RDKit Mol object) or a networkx Graph representing the molecule.

    Yields
    ------
    tuple[int, ...]
        Each tuple represents a main chain fragment as a tuple of atom indices.

    Raises
    ------
    TypeError
        If the input is neither an RDKit Mol object nor a networkx Graph.

    Notes
    -----
    This function identifies the main chain fragments in the molecule or graph. It removes hydrogen atoms
    from RDKit Mol objects and constructs a networkx Graph from the adjacency matrix.

    For each connected component in the graph, it identifies the start and end nodes of the main chain
    using a breadth-first search approach. The main chain fragments are generated as tuples of atom indices
    using the shortest path algorithm in networkx.

    """

    if isinstance(mol_or_graph, Chem.rdchem.Mol):
        mol = Chem.RemoveHs(mol_or_graph)
        graph: nx.Graph = nx.from_numpy_array(
            Chem.GetAdjacencyMatrix(mol),
            # nodelist=range(1, mol.GetNumAtoms() + 1),
        )
    elif isinstance(mol_or_graph, nx.Graph):
        graph = mol_or_graph
    else:
        raise TypeError

    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        start, end = _bfs_farthest_node(
            subgraph,
            ignore_nodes=nodes_in_cycles(subgraph),
            return_length=False,
        )
        yield tuple(nx.shortest_path(subgraph, start, end))
