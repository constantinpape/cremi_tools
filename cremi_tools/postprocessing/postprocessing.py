import numpy as np
import vigra
from scipy.ndimage import convolve
import nifty
import nifty.graph.rag as nrag


def _edges2d(seg):
    gx = convolve(seg + 1, np.array([-1., 1.]).reshape(1, 2))
    gy = convolve(seg + 1, np.array([-1., 1.]).reshape(2, 1))
    return ((gx ** 2 + gy **2) > 0).view('uint8')


# TODO implement
def _edges3d(seg):
    raise NotImplementedError("Not implemented")


def seg2edges(segmentation, only_2d_edges=False):
    ndim = segmentation.ndim
    assert ndim in (2, 3)
    if ndim == 3 and only_2d_edges:
        edges = np.zeros_like(segmentation, dtype='uint8')
        for z in range(edges.shape[0]):
            edges[z] = _edges2d(segmentation[z])
    elif ndim == 2:
        edges = _edges2d(seg)
    else:
        edges = _edges3d(seg)
    return edges


# only support 0 ignore label
def merge_small_segments(segmentation, hmap, size_threshold, ignore_background=False):
    seg_ids, seg_counts = np.unique(segmentation, return_counts=True)
    ignore_segs = seg_ids[seg_counts < size_threshold]
    mask = np.ma.masked_array(segmentation, np.in1d(segmentation, ignore_segs)).mask
    if ignore_background:
        ignore_mask = segmentation == 0
        hmap[ignore_mask] = 1.
        segmentation[ignore_mask] = segmentation.max() + 1
    segmentation[mask] = 0
    segmentation, _ = vigra.analysis.watershedsNew(hmap, seeds=segmentation.astype('uint32'))
    if ignore_background:
        vigra.analysis.relabelConsecutive(segmentation, out=segmentation, start_label=1)
        segmentation[ignore_mask] = 0
    else:
        vigra.analysis.relabelConsecutive(segmentation, out=segmentation, start_label=0, keep_zeros=False)
    return segmentation


def merge_fully_enclosed(segmentation, n_threads=8):
    rag = nrag.gridRag(segmentation, numberOfLabels=int(relabeled.max() + 1),
                       numberOfThreads=n_threads)
    ufd = nifty.ufd.ufd(rag.numberOfNodes)
    nodes = np.unique(segmentation)
    for node in nodes:
        adjacency = [adj for adj in rag.nodeAdjacency(node)]
        if len(adjacency) == 1:
            ufd.merge(node, adjacency[0][0])
    labeling = ufd.elementLabeling()
    return nrag.projectScalarNodeDataToPixels(rag, labeling)


def merge_label(segmentation, merge_id, n_threads=8):
    """
    Merge all instances of a given label id into the surrounding labels.
    """
    merge_map = segmentation == merge_id
    relabeled = vigra.analysis.labelMultiArrayWithBackground(segmentation)
    merge_ids = np.unique(relabeled[merge_map])

    n_labels = int(relabeled.max() + 1)
    rag = nrag.gridRag(relabeled, numberOfLabels=n_labels,
                       numberOfThreads=n_threads)
    fake = np.zeros(rag.shape, dtype='float32')
    edge_sizes = nrag.accumulateEdgeMeanAndLength(rag, fake)[:, 1]

    for merge in merge_ids:
        adjacency = [adj for adj in rag.nodeAdjacency(merge)]
        if len(adjacency) == 1:
            node = adjacency[0][0]
        else:
            node = 0
            size = 0
            for adj in adjacency:
                curr_node, edge = adj
                if edge_sizes[edge] > size and curr_node != 0:
                    node = curr_node
                    size = edge_sizes[edge]
        relabeled[relabeled == merge] = node
    relabeled = vigra.analysis.labelMultiArrayWithBackground(relabeled)
    return relabeled



def merge_mitos(segmentation, mito_map):
    pass
