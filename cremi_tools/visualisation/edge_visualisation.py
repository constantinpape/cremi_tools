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
