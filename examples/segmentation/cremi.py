import sys
import numpy as np

sys.path.append('../..')
import cremi_tools.segmentation as cseg

sys.path.append('/home/papec/Work/my_projects/z5/bld/python')
import z5py

sys.path.append('/home/papec/Work/software/bld/nifty_no_nh5/python')
import nifty
import nifty.graph.rag as nrag


# TODO affinitiy features


def graph_and_bmap_features(path, ws_key, data_key, bb):
    ws = z5py.File(path)[ws_key][bb]
    data_bb = (slice(0, 3),) + bb
    data = np.mean(z5py.File(path)[data_key][data_bb], axis=0)

    extractor = cseg.MeanBoundaryMapFeatures()
    rag, probs, _, edge_sizes = extractor(data, ws)
    return rag, probs, edge_sizes


def multicut_cremi(sample, bb,
                   from_bmaps=True,
                   weight_edges=False):
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sample%s.n5' % sample
    ws_key = 'segmentations/watershed'
    data_key = 'predictions/full_affs'

    # first we calculate the graph and features
    rag, probs, edge_sizes = graph_and_bmap_features(path, ws_key, data_key)

    # next, we convert to multicut costs
    costs = cseg.transform_probabilities_to_costs(probs,
                                                  edge_sizes=edge_sizes if weight_edges else None)
    uv_ids = rag.uvIds()

    # finally, we run multicut
    cutter = cseg.Multicut("kernighan-lin")
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes, rag.uvIds())
    node_labels = cutter(graph, costs)

    segmentation = nrag.projectScalarNodeDataToPixels(rag, node_labels)
    f = z5py.File('./mc_%s.n5' % sample, use_zarr_format=False)
    ds = f.create_dataset('seg0', dtype='uint32', compression='gzip', shape=segmentation.shape, chunks=(64, 64, 64))
    ds[:] = segmentation


if __name__ == '__main__':
    bb = np.s_[:]
    multicut_cremi('A+', bb)
