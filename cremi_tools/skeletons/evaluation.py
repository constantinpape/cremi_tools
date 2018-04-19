import os
from copy import deepcopy

import numpy as np
import nifty.skeletons as nskel
import nifty


def build_skeleton_metrics(label_file, skeleton_file, n_threads=-1):
    assert os.path.exists(label_file), label_file
    assert os.path.exists(skeleton_file), skeleton_file
    skeleton_ids = os.listdir(skeleton_file)
    skeleton_ids = [int(sk) for sk in skeleton_ids if sk.isdigit()]
    skeleton_ids.sort()
    return nskel.SkeletonMetrics(label_file, skeleton_file, skeleton_ids, n_threads)


def load_skeleton_metrics(label_file, skeleton_file, serialization_file):
    assert os.path.exists(label_file), label_file
    assert os.path.exists(skeleton_file), skeleton_file
    assert os.path.exists(serialization_file), serialization_file
    skeleton_ids = os.listdir(skeleton_file)
    skeleton_ids = [int(sk) for sk in skeleton_ids if sk.isdigit()]
    skeleton_ids.sort()
    return nskel.SkeletonMetrics(label_file, skeleton_file, skeleton_ids, serialization_file)


# TODO nifty graph is not really suited here,
# because it assumes dense node labels -> we waste a lot of memory
# go from skeleton edge list to tree
def build_skeleton_tree(edges):
    nodes = np.unique(edges)
    n_nodes = nodes[-1] + 1
    tree = nifty.graph.undirectedGraph(n_nodes)
    tree.insertEdges(edges)
    return tree


def filter_edges(edges, node_list):
    # first check for a priori invalid edges
    valid_edges = (edges != -1).all(axis=1)
    edges = edges[valid_edges]
    # next check for edges with nodes that are not in
    # the node list
    valid_edges = np.in1d(edges, node_list).reshape(edges.shape).all(axis=1)
    return edges[valid_edges]


def get_node_degree(tree, node_id):
    node_degree = len([_ for _ in tree.nodeAdjacency(node_id)])
    return node_degree


def walk_tree(tree, node_id, last_node_id, node_list, max_counter):
    node_degree = get_node_degree(tree, node_id)
    # we stop walking the tree for branching or end points
    if node_degree != 2:
        return
    # find the adjacent nodes and the correct next node by checking for the last node
    adjacency = tree.nodeAdjacency(node_id)
    # TODO check that we actually have a next node and that last node is adjacent ?!
    for adj in adjacency:
        if adj[0] != last_node_id:
            next_node = adj[0]
    node_list.append(next_node)
    # check if we have exceeded the walking depth
    if len(node_list) > max_counter:
        return
    # otherwise go to the next node
    walk_tree(tree, next_node, node_id, node_list, max_counter)


def smooth_distance_statistics(distance_statistics, skeleton_edges, sliding_window_size):
    skeleton_ids = list(distance_statistics.keys())
    skeleton_ids.sort()

    ids2 = list(skeleton_edges.keys())
    ids2.sort()
    assert skeleton_ids == ids2

    smoothed_statistics = {}
    for skel_id, stats in distance_statistics.items():

        skel_statistics = {}

        edges = skeleton_edges[skel_id]
        edges = filter_edges(edges, np.array(list(stats.keys())))
        tree = build_skeleton_tree(edges)

        for node_id in stats:

            values = deepcopy(stats[node_id])
            node_degree = get_node_degree(tree, node_id)

            # we don't smooth for branching points TODO ?!
            if node_degree > 2 or node_degree == 0:
                skel_statistics[node_id] = values
            # we can only go into single direction for an end point
            elif node_degree == 1:
                next_node = next(tree.nodeAdjacency(node_id))[0]
                node_list = []
                walk_tree(tree, next_node, node_id, node_list, sliding_window_size)
            # we walk the tree in both directions for a normal skeleton point
            else:
                adj = tree.nodeAdjacency(node_id)
                left_node = next(adj)[0]
                right_node = next(adj)[0]
                node_list_left = []
                walk_tree(tree, left_node, node_id, node_list_left, sliding_window_size)
                node_list_right = []
                walk_tree(tree, right_node, node_id, node_list_right, sliding_window_size)
                node_list = node_list_left + node_list_right

            # extend the values by the additional nodes we have found
            for node in node_list:
                values.extend(stats[node])
            skel_statistics[node_id] = values

        smoothed_statistics[skel_id] = skel_statistics

    return smoothed_statistics


def bfs(node_id, tree, tree_depths, max_depth, node_list):
    curr_depth = tree_depths[node_id]
    # check if we will exceed the max-depth
    if curr_depth + 1 > max_depth:
        return
    for adj in tree.nodeAdjacency(node_id):
        next_node = adj[0]
        if next_node not in tree_depths:
            tree_depths[next_node] = curr_depth + 1
            node_list.append(next_node)
            bfs(next_node, tree, tree_depths, max_depth, node_list)


def smooth_distance_statistics_bfs(distance_statistics, skeleton_edges, smooth_depth):
    skeleton_ids = list(distance_statistics.keys())
    skeleton_ids.sort()

    ids2 = list(skeleton_edges.keys())
    ids2.sort()
    assert skeleton_ids == ids2

    smoothed_statistics = {}
    for skel_id, stats in distance_statistics.items():

        skel_statistics = {}

        edges = skeleton_edges[skel_id]
        edges = filter_edges(edges, np.array(list(stats.keys())))
        tree = build_skeleton_tree(edges)

        for node_id in stats:
            values = deepcopy(stats[node_id])
            # having tree_depths and node_list is a bit redundant,
            # but I will leave it for convenience for now
            tree_depths = {node_id: 0}
            node_list = []
            bfs(node_id, tree, tree_depths, smooth_depth, node_list)

            # extend the values by the additional nodes we have found
            for node in node_list:
                values.extend(stats[node])
            skel_statistics[node_id] = values

        smoothed_statistics[skel_id] = skel_statistics

    return smoothed_statistics
