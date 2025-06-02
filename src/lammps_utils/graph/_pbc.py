import networkx as nx
import numpy as np
from numpy.typing import ArrayLike


def unwrap_molecule_under_pbc(
    graph: nx.Graph, positions: np.ndarray, cell_size: ArrayLike
) -> np.ndarray:
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
