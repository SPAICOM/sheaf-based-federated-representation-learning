"""
Graph generation utilities for federated learning communication topologies.

This module provides functions to generate various graph structures that define
which agents can communicate and share model updates in federated learning.
"""

from typing import Literal

import networkx as nx


def generate_neighbors(
    mode: Literal['erdos_renyi', 'barabasi', 'fully_connected', 'manual'],
    n_agents: int,
    seed: int = 42,
    p: float = 0.3,
    m: int = 3,
    manual: dict | None = None,
) -> dict[int, set[int]]:
    """Generate neighbor dictionary from a graph model.

    Creates a communication graph for federated learning where edges define
    which agents can share model updates with each other.

    Parameters
    ----------
    mode : str
        Graph generation mode. Options:
        - "manual": Use provided manual dictionary (for custom topologies)
        - "fully_connected": Complete graph (all agents can communicate)
        - "erdos_renyi": Random graph with edge probability p (G(n,p) model)
        - "barabasi": Barabasi-Albert preferential attachment (scale-free)
    n_agents : int
        Number of agents (nodes) in the graph
    seed : int
        Random seed for reproducibility (default: 42)
    p : float
        Edge probability for Erdos-Renyi (default: 0.3)
        Higher p → more edges → denser communication
    m : int
        Number of edges to attach from a new node in Barabasi-Albert
        (default: 3). Controls the power-law exponent.
    manual : dict, optional
        Manual neighbor dictionary, only used when mode="manual"

    Returns
    -------
    dict[int, set[int]]
        Dictionary mapping each node index to the set of its neighbors

    Raises
    ------
    ValueError
        If mode is unknown

    Graph Characteristics
    ----------------------
    - Erdos-Renyi: Random structure, approximate uniform degree distribution
    - Barabasi-Albert: Scale-free, few hubs with many connections
    - Fully Connected: Maximum communication overhead, fastest consensus
    """
    match mode:
        # Mode 1: Manual - use pre-defined neighbor dictionary
        case 'manual':
            if manual is None:
                return {}
            # Convert keys to int and values to sets
            return {int(k): set(v) for k, v in manual.items()}

        # Mode 2: Fully Connected - every agent connects to every other
        case 'fully_connected':
            G = nx.complete_graph(n_agents)

        # Mode 3: Erdos-Renyi random graph
        # Each edge exists independently with probability p
        case 'erdos_renyi':
            G = nx.erdos_renyi_graph(n_agents, p, seed=seed)

        # Mode 4: Barabasi-Albert scale-free graph
        # New nodes attach to existing hubs (preferential attachment)
        case 'barabasi':
            G = nx.barabasi_albert_graph(n_agents, m, seed=seed)

        case _:
            raise ValueError(
                f'Unknown neighbors_mode: {mode}. Valid options: '
                "'manual', 'erdos_renyi', 'barabasi', 'fully_connected'"
            )

    # Convert NetworkX graph to neighbor dictionary format
    # {node: {neighbor1, neighbor2, ...}}
    return {i: set(G.neighbors(i)) for i in range(n_agents)}
