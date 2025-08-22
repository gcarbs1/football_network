"""Utilities for analyzing football networks with persistent homology.

This script demonstrates how to compute persistent homology on a complex
network using the ripser library. It converts a ``networkx`` graph into a
pairwise distance matrix based on shortest path lengths, then feeds the
resulting distances to ``ripser`` to obtain persistence diagrams. A simple
example graph is provided when the script is executed directly.

Dependencies
------------
- networkx
- ripser
- persim
- matplotlib
- numpy

Because these packages are not part of the standard library, they must be
installed separately before running the script.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt


def graph_to_distance_matrix(graph: nx.Graph) -> np.ndarray:
    """Return a dense matrix of shortest path lengths for ``graph``.

    Parameters
    ----------
    graph:
        The input ``networkx`` graph. The graph should be connected; otherwise
        infinite distances will appear between disconnected components.

    Returns
    -------
    numpy.ndarray
        A square matrix where entry ``(i, j)`` contains the shortest path
        length between node ``i`` and node ``j``.
    """

    nodes = list(graph.nodes)
    size = len(nodes)
    dist = np.full((size, size), np.inf, dtype=float)
    for i, u in enumerate(nodes):
        dist[i, i] = 0.0
        lengths = nx.single_source_shortest_path_length(graph, u)
        for v, d in lengths.items():
            j = nodes.index(v)
            dist[i, j] = float(d)
    return dist


def persistent_homology(graph: nx.Graph):
    """Compute persistence diagrams of ``graph`` using its clique complex."""
    dist = graph_to_distance_matrix(graph)
    n = dist.shape[0]
    condensed = dist[np.triu_indices(n, k=1)]
    return ripser(condensed, distance_matrix=True)["dgms"]


def example_graph(num_nodes: int = 10, p: float = 0.4) -> nx.Graph:
    """Generate a connected Erdős–Rényi graph for demonstration."""
    graph = nx.erdos_renyi_graph(num_nodes, p, seed=0)
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(num_nodes, p)
    return graph


def plot_diagrams_from_graph(graph: nx.Graph) -> None:
    """Plot persistence diagrams for ``graph``."""
    diagrams = persistent_homology(graph)
    plot_diagrams(diagrams, show=True)


if __name__ == "__main__":
    demo_graph = example_graph()
    plot_diagrams_from_graph(demo_graph)
