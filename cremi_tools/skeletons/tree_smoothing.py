from copy import deepcopy
import numpy as np
import nifty


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


def walk_tree(tree, node_id, last_node_id, node_list, max_depth):
    # check if we have exceeded the walking depth
    if len(node_list) >= max_depth:
        return
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
    # otherwise go to the next node
    walk_tree(tree, next_node, node_id, node_list, max_depth)


def neighbors_in_window(tree, node_id, max_depth):
    """
    Find the neighbors in the window around 'node_id'.
    Returns list of neighbor and their distance to the node.
    """
    node_degree = get_node_degree(tree, node_id)
    # branching node (or invalid node) -> we don't smooth
    if node_degree > 2 or node_degree == 0:
        return [], []
    # terminal node -> can apply sliding window only along single direction
    elif node_degree == 1:
        next_node = next(tree.nodeAdjacency(node_id))[0]
        node_list = [next_node]
        walk_tree(tree, next_node, node_id, node_list, max_depth)
        tree_depths = list(range(1, len(node_list) + 1))
    # normal tree node -> apply sliding window along both directions
    else:
        adj = tree.nodeAdjacency(node_id)
        left_node = next(adj)[0]
        right_node = next(adj)[0]
        node_list_left = [left_node]
        walk_tree(tree, left_node, node_id, node_list_left, max_depth)
        node_list_right = [right_node]
        walk_tree(tree, right_node, node_id, node_list_right, max_depth)
        tree_depths = list(range(1, len(node_list_left) + 1)) + list(range(1, len(node_list_right) + 1))
        node_list = node_list_left + node_list_right
    return node_list, tree_depths


def bfs(tree, node_id, max_depth, tree_depths):
    curr_depth = tree_depths[node_id]
    # check if we will exceed the max-depth
    if curr_depth + 1 > max_depth:
        return
    for adj in tree.nodeAdjacency(node_id):
        next_node = adj[0]
        if next_node not in tree_depths:
            tree_depths[next_node] = curr_depth + 1
            bfs(tree, next_node, max_depth, tree_depths)


def neighbors_bfs(tree, node_id, max_depth):
    """
    Find the neighbors within radius 'max_depth' around 'node_id'.
    Returns list of neighbor and their distance to the node.
    """
    tree_depths = {node_id: 0}
    bfs(tree, node_id, max_depth, tree_depths)
    node_list = [node for node, depth in tree_depths.items() if depth > 0]
    depths = [tree_depths[node] for node in node_list]
    return node_list, depths


def smooth_along_tree(tree_values, tree_edges, max_depth, neighbor_function):
    assert callable(neighbor_function)

    smoothed_values = {}
    edges = filter_edges(tree_edges, np.array(list(tree_values.keys())))
    tree = build_skeleton_tree(edges)

    # iterate over the tree nodes and associated values
    for node_id in tree_values:

        # extract the values of this node
        values = deepcopy(tree_values[node_id])
        # find the nodes in the neighborhood according to the neighbor function
        # and the maximal search depth
        node_list, _ = neighbor_function(tree, node_id, max_depth)

        # extend the values by values of the additional nodes we have found
        for node in node_list:
            values.extend(tree_values[node])
        smoothed_values[node_id] = values

    return smoothed_values


def smooth_sliding_window(tree_values, tree_edges, max_depth):
    return smooth_along_tree(tree_values, tree_edges, max_depth, neighbors_in_window)


def smooth_bfs(tree_values, tree_edges, max_depth):
    return smooth_along_tree(tree_values, tree_edges, max_depth, neighbors_bfs)
