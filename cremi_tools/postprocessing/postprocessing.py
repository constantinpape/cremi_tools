import numpy as np
import vigra
import nifty
import nifty.graph.rag as nrag


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
    rag = nrag.gridRag(segmentation, numberOfThreads=n_threads)
    ufd = nifty.ufd.ufd(rag.numberOfNodes)
    nodes = np.unique(segmentation)
    for node in nodes:
        adjacency = [adj for adj in rag.nodeAdjacency(node)]
        if len(adjacency) == 1:
            ufd.merge(node, adjacency[0][0])
    labeling = ufd.elementLabeling()
    return nrag.projectScalarNodeDataToPixels(rag, labeling)


def merge_mitos(segmentation, mito_map):
    pass
