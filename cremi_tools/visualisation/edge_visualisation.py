import numpy as np
import nifty.graph.rag as nrag


def visualize_probabilities(rag, probs, edge_direction=2):
    assert rag.numberOfEdges == len(probs), "%i, %i" % (rag.numberOfEdges, len(probs))

    edge_builder = nrag.ragCoordinates(rag)
    edge_map_att = np.zeros_like(probs, dtype='uint32')
    edge_map_rep = np.zeros_like(probs, dtype='uint32')

    # build attractive edges
    edge_map_att[probs <= .1] = 3
    edge_map_att[np.logical_and(probs > .1, probs <= .3)] = 2
    edge_map_att[np.logical_and(probs > .3, probs <= .5)] = 1
    edge_vol_att = edge_builder.edgesToVolume(edge_map_att, edgeDirection=edge_direction)

    # build repulsive edges
    edge_map_rep[np.logical_and(probs > .5, probs <= .7)] = 1
    edge_map_rep[np.logical_and(probs > .7, probs <= .9)] = 2
    edge_map_rep[probs > .9] = 3
    edge_vol_rep = edge_builder.edgesToVolume(edge_map_rep, edgeDirection=edge_direction)

    # build edge ids
    edge_ids = np.arange(rag.numberOfEdges).astype('uint32')
    edge_id_vol = edge_builder.edgesToVolume(edge_ids, edgeDirection=edge_direction)
    return edge_id_vol, edge_vol_att, edge_vol_rep


# this returns a 2d array with the all the indices of matching rows for a and b
# cf. http://stackoverflow.com/questions/20230384/find-indexes-of-matching-rows-in-two-2-d-arrays
def find_matching_row_indices(x, y):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # using a dictionary, this is faster than the pure np variant
    indices = []
    rows_x = {tuple(row): i for i, row in enumerate(x)}
    for i, row in enumerate(y):
        if tuple(row) in rows_x:
            indices.append([rows_x[tuple(row)], i])
    return np.array(indices)


def visualize_probabilities_for_subvolume(sub_ws, probs, uv_ids, edge_direction=2):
    rag = nrag.gridRag(sub_ws, numberOfLabels=int(sub_ws.max()) + 1)

    edge_builder = nrag.ragCoordinates(rag)
    edge_map_att = np.zeros(rag.numberOfEdges, dtype='uint32')
    edge_map_rep = np.zeros(rag.numberOfEdges, dtype='uint32')

    rag_uvs = rag.uvIds()
    indices = find_matching_row_indices(rag_uvs, uv_ids)[:, 0]

    sub_probs = probs[indices]

    # build attractive edges
    edge_map_att[sub_probs <= .1] = 3
    edge_map_att[np.logical_and(sub_probs > .1, sub_probs <= .3)] = 2
    edge_map_att[np.logical_and(sub_probs > .3, sub_probs <= .5)] = 1
    edge_vol_att = edge_builder.edgesToVolume(edge_map_att, edgeDirection=edge_direction)

    # build repulsive edges
    edge_map_rep[np.logical_and(sub_probs > .5, sub_probs <= .7)] = 1
    edge_map_rep[np.logical_and(sub_probs > .7, sub_probs <= .9)] = 2
    edge_map_rep[sub_probs > .9] = 3
    edge_vol_rep = edge_builder.edgesToVolume(edge_map_rep, edgeDirection=edge_direction)

    # build edge ids
    edge_ids = np.arange(rag.numberOfEdges).astype('uint32')
    edge_id_vol = edge_builder.edgesToVolume(edge_ids, edgeDirection=edge_direction)
    return edge_id_vol, edge_vol_att, edge_vol_rep
