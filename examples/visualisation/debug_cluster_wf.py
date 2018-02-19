import sys
import numpy as np

import nifty
import nifty.graph.rag as nrag

sys.path.append('../..')
import cremi_tools.segmentation as cseg
from cremi_tools.viewer.volumina import view

sys.path.append('/home/papec/Work/my_projects/z5/bld/python')
import z5py


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


def check_graph_blocks():
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sampleA+.n5'
    ws_key = 'segmentations/watershed'

    ws = z5py.File(path)[ws_key][:].astype('uint32')
    rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1)
    uv_ids = rag.uvIds()

    n_blocks = 1680
    graph_path = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/cremi_A+/tmp_files/graph.n5/sub_graphs/s0'

    for block_id in range(n_blocks):
        graph_ds = z5py.File(graph_path)['block_%i' % block_id]
        if 'edges' not in graph_ds:
            continue
        print("Checking block", block_id)
        edges = graph_ds['edges'][:]
        edge_ids = graph_ds['edgeIds'][:]
        assert len(edges) == len(edge_ids)
        # rag_edge_ids = find_matching_row_indices(edges, uv_ids)[:, 0]
        # assert len(edge_ids) == len(rag_edge_ids), "%i, %i" % len(edge_ids) == len(rag_edge_ids)
        # for e, ei in zip(edges, edge_ids):
        #     print(e, uv_ids[ei])
        # assert (rag_edge_ids == edge_ids).all()
        assert (edges == uv_ids[edge_ids]).all()
    print("All passed")


def check_feats():
    print("Zero-entries for whole features")
    feature_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cache/cremi_A+/tmp_files/features.n5'
    feat = z5py.File(feature_path)['features'][:, 0:1]
    zero_entries = np.isclose(feat, 0)
    print(np.sum(zero_entries), '/')
    print(len(zero_entries))

    n_blocks = 1680
    graph_path = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/cremi_A+/tmp_files/graph.n5/sub_graphs/s0'
    feature_pre = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cache/cremi_A+/tmp_files/features.n5/blocks'

    feat_from_blocks = np.zeros_like(feat, dtype='float64')
    for block_id in range(n_blocks):
        graph_ds = z5py.File(graph_path)['block_%i' % block_id]
        if 'edgeIds' not in graph_ds:
            continue
        feats = z5py.File(feature_pre)['block_%i' % block_id][:, 0:1]
        edge_ids = graph_ds['edgeIds'][:]
        assert len(feats) == len(edge_ids)
        feat_from_blocks[edge_ids] += feats

    print("Zero-entries for block-accumulated features:")
    zero_blocks = np.isclose(feat_from_blocks, 0)
    print(np.sum(zero_blocks), '/')
    print(len(zero_blocks))


def mc_from_costs(sample, out_key=None):
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sampleA+.n5'
    # path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/testdata1.n5'
    ws_key = 'segmentations/watershed'
    # data_key = 'predictions/full_affs'
    raw_key = 'raw'

    # first we calculate the graph and features
    ws = z5py.File(path)[ws_key][:].astype('uint32')
    rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1)

    feature_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cache/cremi_A+/tmp_files/features.n5'
    # feature_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/aff_features.n5'
    probs = 1. - z5py.File(feature_path)['features'][:, 0:1]
    probs = probs.squeeze()
    assert rag.numberOfEdges == len(probs), "%i, %i" % (rag.numberOfEdges, len(probs))

    costs = cseg.transform_probabilities_to_costs(probs, edge_sizes=None)
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
        raw = z5py.File(path)[raw_key][:]
        view([raw, ws, segmentation])


def edge_costs_full(sample):
    # path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/testdata1.n5'
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sampleA+.n5'
    feature_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cache/cremi_A+/tmp_files/features.n5'
    # feature_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/aff_features.n5'

    ws_key = 'segmentations/watershed'
    # data_key = 'predictions/full_affs'
    raw_key = 'raw'

    # first we calculate the graph and features
    ws = z5py.File(path)[ws_key][:].astype('uint32')
    rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1)
    probs = 1. - z5py.File(feature_path)['features'][:, 0:1]

    edge_id_vol, edge_vol_att, edge_vol_rep = get_edge_costs(rag, probs)

    bb = np.s_[:100, 512:2000, 512:2000]
    raw = z5py.File(path)[raw_key][bb]
    view([raw, ws[bb], edge_id_vol[bb], edge_vol_att[bb], edge_vol_rep[bb]],
         ['raw', 'ws', 'edge-ids', 'attractive edges', 'repulsive edges'],
         layer_types=["Grayscale", "RandomColors", "RandomColors", "Blue", "Red"])


def edge_costs_block(block_id):
    block_path = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/cremi_A+/tmp_files/features.n5/blocks'
    graph_path = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/cremi_A+/tmp_files/graph.n5/sub_graphs/s0'

    graph_ds = z5py.File(graph_path)['block_%i' % block_id]
    roi_begin = graph_ds.attrs['roiBegin']
    roi_end = graph_ds.attrs['roiEnd']
    probs = 1. - z5py.File(block_path)['block_%i' % block_id][:, 0:1]

    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sampleA+.n5'
    ws_key = 'segmentations/watershed'
    raw_key = 'raw'

    bb = tuple(slice(rb, re) for rb, re in zip(roi_begin, roi_end))
    ws = z5py.File(path)[ws_key][bb].astype('uint32')

    rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1)
    edges = graph_ds['edges'][:]
    assert len(edges) == rag.numberOfEdges
    assert (rag.uvIds() == edges).all()

    edge_id_vol, edge_vol_att, edge_vol_rep = get_edge_costs(rag, probs)

    raw = z5py.File(path)[raw_key][bb]
    view([raw, ws, edge_id_vol, edge_vol_att, edge_vol_rep],
         ['raw', 'ws', 'edge-ids', 'attractive edges', 'repulsive edges'],
         layer_types=["Grayscale", "RandomColors", "RandomColors", "Blue", "Red"])


def get_edge_costs(rag, probs):
    assert rag.numberOfEdges == len(probs), "%i, %i" % (rag.numberOfEdges, len(probs))

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
    return edge_id_vol, edge_vol_att, edge_vol_rep


def new_test_data(sample='A+'):
    import vigra
    bb = np.s_[:100, 512:2000, 512:2000]
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sample%s.n5' % sample
    ws_key = 'segmentations/watershed'
    data_key = 'predictions/full_affs'
    raw_key = 'raw'
    f = z5py.File(path)
    path_out = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/testdata1.n5'
    f_out = z5py.File(path_out, use_zarr_format=False)

    for key in [raw_key, ws_key]:
        if key == data_key:
            bb_ = (slice(None),) + bb
            chunks = (3, 25, 200, 200)
        else:
            bb_ = bb
            chunks = (25, 200, 200)
        data = f[key][bb_]
        if key == ws_key:
            data = vigra.analysis.labelVolumeWithBackground(data.astype('uint32'))
            data = data.astype('uint64')
        ds = f_out.create_dataset(key, dtype=data.dtype, compression='gzip', shape=data.shape, chunks=chunks)
        ds[:] = data


if __name__ == '__main__':
    check_feats()
    # check_graph_blocks()
    # block_id = 20
    # edge_costs_block(block_id)
    # new_test_data()
    # edge_costs_full('A+')
    # mc_from_costs('A+', out_key='mc_debug')
