"""
Graph generation utilities for federated learning communication topologies.

This module provides functions to generate various graph structures that define
which agents can communicate and share model updates in federated learning.
"""

from typing import Literal

import networkx as nx


def generate_neighbors(
    mode: Literal[
        'erdos_renyi',
        'barabasi',
        'fully_connected',
        'manual',
        'class_overlap',
    ],
    n_agents: int,
    seed: int = 42,
    p: float = 0.3,
    m: int = 3,
    manual: dict | None = None,
    target_classes: dict[int, set[int]] | None = None,
    max_edge_frac: float = 0.4,
    similarity: Literal['intersection', 'jaccard'] = 'intersection',
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
        - "class_overlap": Sparse, connected graph keeping the edges between
          the agents that share the most target classes (see
          ``class_overlap_neighbors``).
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
    target_classes : dict[int, set[int]], optional
        Per-agent target class sets, required when mode="class_overlap".
    max_edge_frac : float
        Edge budget as a fraction of all possible edges, used when
        mode="class_overlap" (default: 0.4).
    similarity : str
        Edge weight for mode="class_overlap": "intersection" (number of
        shared classes) or "jaccard" (default: "intersection").

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

        # Mode 5: Class-overlap - sparse graph of the most semantically
        # similar agents (those sharing the most target classes).
        case 'class_overlap':
            if target_classes is None:
                raise ValueError(
                    "mode='class_overlap' requires target_classes "
                    '(per-agent target class sets).'
                )
            return class_overlap_neighbors(
                target_classes,
                max_edge_frac=max_edge_frac,
                similarity=similarity,
            )

        case _:
            raise ValueError(
                f'Unknown neighbors_mode: {mode}. Valid options: '
                "'manual', 'erdos_renyi', 'barabasi', 'fully_connected', "
                "'class_overlap'"
            )

    # Convert NetworkX graph to neighbor dictionary format
    # {node: {neighbor1, neighbor2, ...}}
    return {i: set(G.neighbors(i)) for i in range(n_agents)}


def class_overlap_neighbors(
    target_classes: dict[int, set[int]],
    max_edge_frac: float = 0.4,
    similarity: Literal['intersection', 'jaccard'] = 'intersection',
) -> dict[int, set[int]]:
    """Sparse, connected graph linking the agents with the most shared classes.

    Every candidate edge (i, j) is weighted by how many target classes agents
    ``i`` and ``j`` have in common (``intersection``) or by their Jaccard
    similarity. The returned graph keeps the highest-weight edges subject to two
    constraints:

    1. At most ``max_edge_frac`` of all possible ``n(n-1)/2`` edges.
    2. The graph is connected — a maximum-weight spanning tree is always kept as
       the backbone, so agents that share few/no classes still reach the rest of
       the network through their strongest available link. Connectivity takes
       priority: if the edge budget is below ``n - 1`` it is raised to ``n - 1``.

    The result is deterministic (ties broken by ascending node index); no seed
    is needed.

    Parameters
    ----------
    target_classes : dict[int, set[int]]
        Maps each agent id to the set of classes it is skewed toward.
    max_edge_frac : float
        Edge budget as a fraction of all possible edges (default: 0.4).
    similarity : str
        "intersection" (count of shared classes) or "jaccard".

    Returns
    -------
    dict[int, set[int]]
        Symmetric neighbor dictionary, one entry per agent.
    """
    agents = sorted(target_classes)
    n = len(agents)
    neighbors: dict[int, set[int]] = {i: set() for i in agents}
    if n <= 1:
        return neighbors

    def weight(i: int, j: int) -> float:
        inter = len(target_classes[i] & target_classes[j])
        if similarity == 'jaccard':
            union = len(target_classes[i] | target_classes[j])
            return inter / union if union else 0.0
        return float(inter)

    # Complete weighted graph over the agents.
    G = nx.Graph()
    G.add_nodes_from(agents)
    for a in range(n):
        for b in range(a + 1, n):
            i, j = agents[a], agents[b]
            G.add_edge(i, j, weight=weight(i, j))

    total_possible = n * (n - 1) // 2
    budget = int(max_edge_frac * total_possible)  # floor → never exceeds cap
    budget = max(budget, n - 1)  # connectivity needs at least n - 1 edges

    # 1) Connected backbone: maximum-weight spanning tree.
    kept: set[frozenset[int]] = {
        frozenset((i, j)) for i, j in nx.maximum_spanning_tree(G).edges()
    }

    # 2) Fill the remaining budget with the highest-overlap edges.
    ranked = sorted(
        G.edges(data=True),
        key=lambda e: (-e[2]['weight'], min(e[0], e[1]), max(e[0], e[1])),
    )
    for i, j, _ in ranked:
        if len(kept) >= budget:
            break
        kept.add(frozenset((i, j)))

    for edge in kept:
        i, j = tuple(edge)
        neighbors[i].add(j)
        neighbors[j].add(i)

    return neighbors
