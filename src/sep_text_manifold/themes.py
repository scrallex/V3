"""
Theme detection and graph analytics for the Sep Text Manifold.

The functions in this module build a co‑occurrence graph of strings and
apply community detection algorithms to identify themes.  They also
compute simple centrality scores used in the connector score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import networkx as nx
    from networkx.algorithms.approximation import (
        betweenness_centrality as approximate_betweenness_centrality,
    )
except ImportError:  # pragma: no cover
    nx = None  # type: ignore
    approximate_betweenness_centrality = None  # type: ignore


def build_theme_graph(
    string_windows: Dict[str, Iterable[int]],
    *,
    cooccurrence_threshold: int = 1,
    max_members_per_window: int = 100,
    min_pmi: float = 0.0,
    max_degree: Optional[int] = None,
    occurrence_counts: Optional[Dict[str, int]] = None,
    total_windows: Optional[int] = None,
) -> "Graph":
    """Build an undirected co‑occurrence graph from string window coverage.

    Parameters
    ----------
    string_windows:
        Mapping from string to the iterable of manifold window
        identifiers that covered at least one occurrence of the string.
    cooccurrence_threshold:
        Minimum shared window count for an edge to be included.

    Returns
    -------
    Graph
        A NetworkX graph if the `networkx` package is installed;
        otherwise a simple adjacency dictionary.
    """
    window_sets: Dict[str, Set[int]] = {
        string: set(windows)
        for string, windows in string_windows.items()
    }
    nodes = list(window_sets.keys())
    # Build co-occurrence weights by aggregating over windows.
    window_buckets: Dict[int, List[str]] = defaultdict(list)
    for string, windows in window_sets.items():
        for wid in windows:
            window_buckets[wid].append(string)
    edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)
    for members in window_buckets.values():
        if len(members) < 2:
            continue
        unique_members = sorted(set(members))
        if max_members_per_window and len(unique_members) > max_members_per_window:
            unique_members = unique_members[:max_members_per_window]
        for u, v in combinations(unique_members, 2):
            edge_weights[(u, v)] += 1
    total_windows_value = total_windows if total_windows and total_windows > 0 else None
    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for (u, v), weight in edge_weights.items():
            if weight >= cooccurrence_threshold:
                if min_pmi > 0.0 and occurrence_counts and total_windows_value:
                    occ_u = occurrence_counts.get(u)
                    occ_v = occurrence_counts.get(v)
                    if not occ_u or not occ_v:
                        continue
                    numerator = weight * total_windows_value
                    denominator = occ_u * occ_v
                    if denominator == 0:
                        continue
                    pmi = math.log(numerator / denominator)
                    if pmi < min_pmi:
                        continue
                if max_degree is not None and (
                    G.degree(u) >= max_degree or G.degree(v) >= max_degree
                ):
                    continue
                G.add_edge(u, v, weight=weight)
        return G
    else:
        adj: Dict[str, Set[str]] = defaultdict(set)
        for node in nodes:
            adj.setdefault(node, set())
        for (u, v), weight in edge_weights.items():
            if weight >= cooccurrence_threshold:
                if min_pmi > 0.0 and occurrence_counts and total_windows_value:
                    occ_u = occurrence_counts.get(u)
                    occ_v = occurrence_counts.get(v)
                    if not occ_u or not occ_v:
                        continue
                    numerator = weight * total_windows_value
                    denominator = occ_u * occ_v
                    if denominator == 0:
                        continue
                    pmi = math.log(numerator / denominator)
                    if pmi < min_pmi:
                        continue
                if max_degree is not None:
                    if len(adj[u]) >= max_degree or len(adj[v]) >= max_degree:
                        continue
                adj[u].add(v)
                adj[v].add(u)
        return adj


def detect_themes(graph: "Graph") -> List[Set[str]]:
    """Detect communities/themes in the co‑occurrence graph.

    If NetworkX is available, this function uses the greedy modularity
    community algorithm.  Otherwise, it returns a single community
    containing all nodes.

    Returns
    -------
    List[Set[str]]
        A list of sets, each set containing strings belonging to a
        theme.
    """
    if nx is not None and isinstance(graph, nx.Graph):
        try:
            from networkx.algorithms.community import greedy_modularity_communities
        except Exception:
            # Fallback: single community
            return [set(graph.nodes())]
        communities = list(greedy_modularity_communities(graph))
        return [set(c) for c in communities]
    else:
        # Without networkx, treat each connected component in the
        # adjacency dict as a theme.
        if isinstance(graph, dict):
            visited: Set[str] = set()
            components: List[Set[str]] = []
            for node in graph.keys():
                if node in visited:
                    continue
                stack = [node]
                component: Set[str] = set()
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)
                    stack.extend(graph.get(current, []))
                components.append(component)
            return components if components else [set(graph.keys())]
        return []


def compute_graph_metrics(
    graph: "Graph",
    *,
    mode: str = "full",
    betweenness_sample: Optional[int] = None,
    max_full_nodes: Optional[int] = None,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute centrality metrics for each node in the theme graph.

    Parameters
    ----------
    graph:
        The theme graph (either a NetworkX graph or adjacency dict).
    mode:
        One of ``"full"`` (exact betweenness), ``"fast"`` (sampled/
        approximate betweenness) or ``"off"`` (skip betweenness and
        bridging scores).
    betweenness_sample:
        Optional explicit sample size for the approximate betweenness
        computation when ``mode`` is not ``"full"`` or when the graph
        exceeds ``max_full_nodes``.
    max_full_nodes:
        Optional limit on the number of nodes allowed for the exact
        betweenness run.  If the graph is larger, the function falls
        back to the approximate routine even when ``mode="full"``.
    random_seed:
        Seed forwarded to the approximate routine for reproducibility.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Centrality metrics keyed by node identifier.  When NetworkX is
        unavailable the function returns zeroed placeholders so the
        downstream pipeline can proceed.
    """

    metrics: Dict[str, Dict[str, float]] = {}
    # Fast exit when metrics are explicitly disabled.
    if mode.lower() == "off":
        if isinstance(graph, dict):
            node_list = list(graph.keys())
        elif nx is not None and isinstance(graph, nx.Graph):
            node_list = list(graph.nodes())
        else:
            node_list = []
        denominator = max(1, len(node_list) - 1)
        for node in node_list:
            degree = len(graph[node]) if isinstance(graph, dict) else graph.degree(node)
            metrics[node] = {
                "betweenness": 0.0,
                "bridging": 0.0,
                "theme_entropy_neighbors": 0.0,
                "redundant_degree": degree / denominator if denominator else 0.0,
            }
        return metrics

    if nx is not None and isinstance(graph, nx.Graph):
        node_count = graph.number_of_nodes()
        run_exact = mode.lower() == "full"
        if run_exact and max_full_nodes is not None and node_count > max_full_nodes:
            run_exact = False

        if run_exact and betweenness_sample is None:
            betweenness = nx.betweenness_centrality(graph, normalized=True)
        else:
            if approximate_betweenness_centrality is None:
                # Approximation helper missing; fall back to zeros to keep the pipeline moving.
                betweenness = {node: 0.0 for node in graph.nodes()}
            else:
                samples = betweenness_sample
                if samples is None:
                    samples = max(32, min(1024, node_count // 20 or 1))
                betweenness = approximate_betweenness_centrality(
                    graph,
                    k=min(samples, node_count),
                    normalized=True,
                    seed=random_seed,
                )

        bridging = betweenness
        for node in graph.nodes():
            neighbours = list(graph.neighbors(node))
            degs = [graph.degree(v) for v in neighbours]
            total = sum(degs)
            if not neighbours or total == 0:
                entropy = 0.0
            else:
                entropy = 0.0
                for d in degs:
                    p = d / total
                    if p > 0:
                        entropy -= p * math.log2(p)
                if len(neighbours) > 1:
                    entropy /= math.log2(len(neighbours))
                else:
                    entropy = 0.0
            metrics[node] = {
                "betweenness": float(betweenness.get(node, 0.0)),
                "bridging": float(bridging.get(node, 0.0)),
                "theme_entropy_neighbors": entropy,
                "redundant_degree": graph.degree(node) / max(1, node_count - 1),
            }
        return metrics

    # Fallback path when NetworkX is unavailable or a plain adjacency dict is provided.
    if isinstance(graph, dict):
        node_count = len(graph)
        for node, neighbours in graph.items():
            degree = len(neighbours)
            metrics[node] = {
                "betweenness": 0.0,
                "bridging": 0.0,
                "theme_entropy_neighbors": 0.0,
                "redundant_degree": degree / max(1, node_count - 1),
            }
    return metrics
