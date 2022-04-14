from itertools import chain
from operator import itemgetter
from typing import Iterable, List, Mapping, Sequence, Set, Tuple

from genutility.iter import findfirst
from genutility.math import minmax
from genutility.numpy import assert_square
from genutility.search import bisect_right_sequence
from genutility.sequence import triangular
from genutility.statistics import mean


def average_cluster_similarity(
    first: Sequence[int],
    second: Sequence[int],
    similarity_matrix: Mapping[Tuple[int, int], float],
) -> float:

    sum = 0.0
    for a in first:
        for b in second:
            x, y = minmax(a, b)
            sum += similarity_matrix[x, y]

    return sum / (len(first) * len(second))


def similarity_node_cluster(
    node: int,
    cluster: Sequence[int],
    similarity_matrix: Mapping[Tuple[int, int], float],
) -> float:

    return mean([similarity_matrix[minmax(node, n2)] for n2 in cluster])


def compute_singletons(cliques_set: Set[int], sequential_clusters: Sequence[Sequence[int]]) -> List[int]:

    sequential_clusters_set = set(chain.from_iterable(sequential_clusters))
    return sorted(cliques_set - sequential_clusters_set)


def get_sequential_clusters(
    cliques: Sequence[Sequence[int]],
    similarity_matrix: Mapping[Tuple[int, int], float],
    minimum_cluster_size: int,
) -> List[List[int]]:

    sequential_clusters: List[List[int]] = []

    merge_cliques(cliques, sequential_clusters)
    merge_singletons(cliques, sequential_clusters, similarity_matrix)
    merge_small_sequences(sequential_clusters, similarity_matrix, minimum_cluster_size)

    return sequential_clusters


def merge_cliques(cliques: Iterable[Sequence[int]], sequential_clusters: List[List[int]]) -> None:

    change = True

    while change:

        change = False
        for clique in cliques:
            for node_i, node_j in triangular(clique):
                node_1, node_2 = minmax(node_i, node_j)

                _, existingClusterFirst = findfirst(lambda sc: node_1 in sc, sequential_clusters)
                _, existingClusterSecond = findfirst(lambda sc: node_2 in sc, sequential_clusters)

                if (
                    existingClusterFirst is not None and existingClusterSecond is not None
                ):  # Both nodes from the clique already placed in clusters
                    continue

                # Neither of the nodes is in the cluster
                elif existingClusterFirst is None and existingClusterSecond is None:

                    if (node_2 - node_1) == 1:  # if these are consecutive sentences, we make a new cluster
                        new_cluster = [node_1, node_2]
                        pos = bisect_right_sequence(sequential_clusters, new_cluster[-1], itemgetter(0))
                        sequential_clusters.insert(pos, new_cluster)

                        change = True

                else:  # one node is in one cluster, the other isn't
                    if existingClusterFirst is not None:
                        cluster = existingClusterFirst
                        node = node_2
                    else:
                        cluster = existingClusterSecond
                        node = node_1

                    if (node == cluster[0] - 1) or (node == cluster[-1] + 1):
                        cluster.append(node)
                        cluster.sort()

                        change = True


def merge_small_sequences(
    sequential_clusters: List[List[int]],
    similarity_matrix: Mapping[Tuple[int, int], float],
    minimum_cluster_size: int = 1,
) -> None:

    while True:

        i, _ = findfirst(
            lambda sc: len(sc) < minimum_cluster_size, sequential_clusters
        )  # note: breaking change to original here

        if i is None:
            break

        if i == 0:  # first cluster
            similarity_prev = 0.0
        else:
            similarity_prev = average_cluster_similarity(
                sequential_clusters[i - 1], sequential_clusters[i], similarity_matrix
            )

        if i == (len(sequential_clusters) - 1):  # last cluster
            similarity_next = 0.0
        else:
            similarity_next = average_cluster_similarity(
                sequential_clusters[i], sequential_clusters[i + 1], similarity_matrix
            )

        if similarity_prev > similarity_next:
            merge_ind = i - 1
        else:
            merge_ind = i + 1

        sequential_clusters[i] = sorted(sequential_clusters[merge_ind] + sequential_clusters[i])
        del sequential_clusters[merge_ind]


def merge_singletons(
    cliques: Iterable[Sequence[int]],
    sequential_clusters: List[List[int]],
    similarity_matrix: Mapping[Tuple[int, int], float],
) -> None:

    assert_square("similarity_matrix", similarity_matrix)

    clique_nodes_set = set(chain.from_iterable(cliques))
    singletons = compute_singletons(clique_nodes_set, sequential_clusters)
    dim, dim2 = similarity_matrix.shape  # fixme: cannot be generic mapping here

    while singletons:

        node = singletons[0]
        _, previousNodeCluster = findfirst(lambda sc: node - 1 in sc, sequential_clusters)
        _, nextNodeCluster = findfirst(lambda sc: node + 1 in sc, sequential_clusters)

        if node == 0:  # first node
            similarity_prev = -1.0
        else:
            if previousNodeCluster is not None:
                similarity_prev = similarity_node_cluster(node, previousNodeCluster, similarity_matrix)
            else:
                similarity_prev = similarity_matrix[node - 1, node]

        if node == dim - 1:  # last node
            similarity_next = -1.0
        else:
            if nextNodeCluster is not None:
                similarity_next = similarity_node_cluster(node, nextNodeCluster, similarity_matrix)
            else:
                similarity_next = similarity_matrix[node, node + 1]

        previous = similarity_prev >= similarity_next
        if previous:
            mergeWithCluster = previousNodeCluster is not None
        else:
            mergeWithCluster = nextNodeCluster is not None

        if mergeWithCluster:
            if previous:
                assert previousNodeCluster  # mypy flow analysis fails otherwise
                previousNodeCluster.append(node)
            else:
                assert nextNodeCluster  # mypy flow analysis fails otherwise
                nextNodeCluster.insert(0, node)

        else:
            new_cluster = []
            if previous:
                new_cluster.append(node - 1)
                new_cluster.append(node)
            else:
                new_cluster.append(node)
                new_cluster.append(node + 1)

            pos = bisect_right_sequence(sequential_clusters, new_cluster[-1], itemgetter(0))
            sequential_clusters.insert(pos, new_cluster)

        singletons = compute_singletons(clique_nodes_set, sequential_clusters)
