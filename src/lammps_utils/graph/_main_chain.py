from collections.abc import Generator
from typing import Union

import networkx as nx


def nodes_in_cycles(graph: nx.Graph) -> set[int]:
    """Returns nodes that belong to cycles in the graph.

    Parameters
    ----------
    graph : nx.Graph
        The input undirected graph.

    Returns
    -------
    set[int]
        A set of node identifiers that belong to cycles in the graph.
    """
    bridges = set(nx.bridges(graph))

    non_bridge_edges = tuple(
        edge
        for edge in graph.edges
        if edge not in bridges and edge[::-1] not in bridges
    )
    subgraph = nx.Graph()
    subgraph.add_edges_from(non_bridge_edges)

    return set(subgraph.nodes)


def _bfs_farthest_node(
    graph: nx.Graph,
    ignore_nodes: set[int] = set(),
    return_length: bool = False,
) -> Union[tuple[int, int], tuple[tuple[int, int], int]]:
    """Performs BFS to find the farthest node from any starting node in the graph.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    ignore_nodes : set[int], optional
        Set of nodes to ignore during the BFS, by default an empty set.
    return_length : bool, optional
        Whether to return the maximum distance along with the node pair, by default False.

    Returns
    -------
    Union[tuple[int, int], tuple[tuple[int, int], int]]
        If `return_length` is False, returns a tuple of the farthest node pair.
        If `return_length` is True, returns a tuple containing the farthest node pair and the maximum distance.

    Raises
    ------
    ValueError
        If no valid pair is found.
    """
    use_nodes: set[int] = set(graph.nodes) - ignore_nodes

    max_pair = None
    max_dist = -1
    for start in use_nodes:
        dict_distances: dict[int, int] = nx.single_source_shortest_path_length(
            graph, start
        )

        for node, dist in sorted(
            dict_distances.items(), key=lambda x: x[1], reverse=True
        ):
            if node in ignore_nodes:
                continue

            if dist > max_dist:
                max_dist = dist
                max_pair = (start, node)

    if max_pair is None:
        raise ValueError("No valid pair found.")

    if return_length:
        return max_pair, max_dist
    else:
        return max_pair


def farthest_node_pair(
    graph: nx.Graph,
    ignore_nodes: set[int] = set(),
    return_length: bool = False,
) -> Union[
    Generator[tuple[int, int], None, None],
    Generator[tuple[tuple[int, int], int], None, None],
]:
    """Finds the farthest node pair in each connected component of the graph.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    ignore_nodes : set[int], optional
        Set of nodes to ignore during the search, by default an empty set.
    return_length : bool, optional
        Whether to return the maximum distance along with the node pair, by default False.

    Yields
    ------
    Union[
        Generator[tuple[int, int], None, None],
        Generator[tuple[tuple[int, int], int], None, None]
    ]
        Yields tuples of farthest node pairs. If `return_length` is True,
        the second value in the tuple is the maximum distance.

    Raises
    ------
    ValueError
        If no valid pair is found in any connected component.
    """
    for component in nx.connected_components(graph):
        yield _bfs_farthest_node(
            graph.subgraph(component),
            ignore_nodes=ignore_nodes,
            return_length=return_length,
        )
