import sys
import numpy as np

import nifty
import nifty.graph.rag as nrag

sys.path.append('../..')
import cremi_tools.segmentation as cseg
from cremi_tools.viewer.volumina import view

sys.path.append('/home/papec/Work/my_projects/z5/bld/python')
import z5py


def graph_and_nn_aff_features(path, ws, data_key, bb):
    data_bb = (slice(0, 3),) + bb
    data = 1. - z5py.File(path)[data_key][data_bb]

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    extractor = cseg.MeanAffinitiyMapFeatures(offsets)
    rag, probs, _, edge_sizes = extractor(data, ws)
    return rag, probs, edge_sizes, data


def graph_and_full_aff_features(path, ws, data_key, bb):
    data_bb = (slice(None),) + bb
    data = 1. - z5py.File(path)[data_key][data_bb]

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    extractor = cseg.MeanAffinitiyMapFeatures(offsets)
    rag, probs, _, edge_sizes = extractor(data, ws)
    return rag, probs, edge_sizes, data


def graph_and_bmap_features(path, ws, data_key, bb):
    data_bb = (slice(0, 3),) + bb
    data = 1. - np.mean(z5py.File(path)[data_key][data_bb], axis=0)

    extractor = cseg.MeanBoundaryMapFeatures()
    rag, probs, _, edge_sizes = extractor(data, ws)
    return rag, probs, edge_sizes, data


def view_result(path, segmentation, ws, data, bb):
    raw = z5py.File(path)['raw'][bb]
    if data.ndim == 4:
        data = data.transpose((1, 2, 3, 0))
    view([raw, ws, segmentation, data])


def mala_cremi(sample, bb,
               from_bmaps=True,
               weight_edges=False,
               out_key=None):
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sample%s.n5' % sample
    ws_key = 'segmentations/watershed'
    data_key = 'predictions/full_affs'

    # first we calculate the graph and features
    ws = z5py.File(path)[ws_key][bb].astype('uint32')
    rag, probs, edge_sizes, data = graph_and_bmap_features(path, ws, data_key, bb)

    # next, we convert to multicut costs
    # costs = cseg.transform_probabilities_to_costs(probs,
    #                                               edge_sizes=edge_sizes if weight_edges else None)
    costs = probs
    uv_ids = rag.uvIds()
    ignore_edges = (uv_ids == 0).any(axis=1)
    # costs[ignore_edges] = 5 * costs.min()
    costs[ignore_edges] = 1.

    # finally, we run multicut
    # cutter = cseg.Multicut("kernighan-lin")
    cutter = cseg.MalaClustering(0.3)

    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(uv_ids)
    node_labels = cutter(graph, costs)

    segmentation = nrag.projectScalarNodeDataToPixels(rag, node_labels)
    if out_key is not None:
        f = z5py.File('./mala_%s.n5' % sample, use_zarr_format=False)
        ds = f.create_dataset(out_key,
                              dtype='uint32',
                              compression='gzip',
                              shape=segmentation.shape,
                              chunks=(64, 64, 64))
        ds[:] = segmentation
    else:
        view_result(path, segmentation, ws, data, bb)


def multicut_cremi(sample, bb,
                   from_nn=False,
                   weight_edges=False,
                   out_key=None):
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sample%s.n5' % sample
    ws_key = 'segmentations/watershed'
    data_key = 'predictions/full_affs'

    # first we calculate the graph and features
    ws = z5py.File(path)[ws_key][bb].astype('uint32')
    if from_nn:
        rag, probs, edge_sizes, data = graph_and_nn_aff_features(path, ws, data_key, bb)
    else:
        rag, probs, edge_sizes, data = graph_and_full_aff_features(path, ws, data_key, bb)

    # next, we convert to multicut costs
    costs = cseg.transform_probabilities_to_costs(probs,
                                                  edge_sizes=edge_sizes if weight_edges else None)
    uv_ids = rag.uvIds()
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = 5 * costs.min()

    # finally, we run multicut
    cutter = cseg.Multicut("kernighan-lin")

    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(uv_ids)
    node_labels = cutter(graph, costs)

    segmentation = nrag.projectScalarNodeDataToPixels(rag, node_labels)
    if out_key is not None:
        f = z5py.File('./mc_%s.n5' % sample, use_zarr_format=False)
        ds = f.create_dataset(out_key,
                              dtype='uint32',
                              compression='gzip',
                              shape=segmentation.shape,
                              chunks=(64, 64, 64))
        ds[:] = segmentation.astype('uint32')
    else:
        view_result(path, segmentation, ws, data, bb)


def view_edge_costs(sample, bb, from_bmaps=False):
    path = '/home/papec/work/neurodata_hdd/ntwrk_papec/cremi_warped/sample%s.n5' % sample
    ws_key = 'segmentations/watershed'
    data_key = 'predictions/full_affs'

    # first we calculate the graph and features
    ws = z5py.File(path)[ws_key][bb].astype('uint32')

    # TODO long-range affinities
    if from_bmaps:
        rag, probs, edge_sizes, data = graph_and_bmap_features(path, ws, data_key, bb)
    else:
        rag, probs, edge_sizes, data = graph_and_nn_aff_features(path, ws, data_key, bb)

    edge_builder = nrag.ragCoordinates(rag)
    edge_map_att = np.zeros_like(probs, dtype='uint32')
    edge_map_rep = np.zeros_like(probs, dtype='uint32')

    # build attractive edges
    edge_map_att[probs <= .1] = 3
    edge_map_att[np.logical_and(probs > .1, probs <= .3)] = 2
    edge_map_att[np.logical_and(probs > .3, probs <= .5)] = 1
    edge_vol_att = edge_builder.edgesToVolume(edge_map_att, edgeDirection=2)

    # build repulsive edges
    edge_map_rep[np.logical_and(probs > .5, probs <= .7)] = 1
    edge_map_rep[np.logical_and(probs > .7, probs <= .9)] = 2
    edge_map_rep[probs > .9] = 3
    edge_vol_rep = edge_builder.edgesToVolume(edge_map_rep, edgeDirection=2)

    # build edge ids
    edge_ids = np.arange(rag.numberOfEdges).astype('uint32')
    edge_id_vol = edge_builder.edgesToVolume(edge_ids, edgeDirection=2)

    if data.ndim == 4:
        data = data.transpose((1, 2, 3, 0))
    view([data, ws, edge_id_vol, edge_vol_att, edge_vol_rep],
         layer_types=["Grayscale", "RandomColors", "RandomColors", "Blue", "Red"])


if __name__ == '__main__':
    # bb = np.s_[:100, 512:2000, 512:2000]
    bb = np.s_[:100, 512:1024, 512:1024]
    multicut_cremi('A+', bb, weight_edges=False, out_key='mcseg_fullaffs_noweight')
    # view_edge_costs('A+', bb, False)
