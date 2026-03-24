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
    """
    Generate neighbor dictionary from a graph model.

    Parameters
    ----------
    mode : str
        Graph generation mode. Options:
        - "manual": Use provided manual dictionary
        - "fully_connected": Complete graph
        - "erdos_renyi": Random graph with edge probability p
        - "barabasi": Barabasi-Albert preferential attachment
          with m edges per new node
    n_agents : int
        Number of agents (nodes) in the graph
    seed : int
        Random seed for reproducibility
    p : float
        Edge probability for Erdos-Renyi (default: 0.3)
    m : int
        Number of edges to attach from a new node in Barabasi-Albert
        (default: 3)
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
    """
    match mode:
        case 'manual':
            if manual is None:
                return {}
            return {int(k): set(v) for k, v in manual.items()}

        case 'fully_connected':
            G = nx.complete_graph(n_agents)

        case 'erdos_renyi':
            G = nx.erdos_renyi_graph(n_agents, p, seed=seed)

        case 'barabasi':
            G = nx.barabasi_albert_graph(n_agents, m, seed=seed)

        case _:
            raise ValueError(
                f'Unknown neighbors_mode: {mode}. Valid options: '
                "'manual', 'erdos_renyi', 'barabasi', 'fully_connected'"
            )

    return {i: set(G.neighbors(i)) for i in range(n_agents)}
