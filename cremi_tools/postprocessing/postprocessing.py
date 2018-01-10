import numpy as np
import vigra
import nifty.graph.rag as nrag


def merge_small_segments(segmentation, hmap, size_threshold):
    _, seg_counts = np.unique(segmentation, return_counts=True)
    ignore_segs = seg_counts < size_threshold
    # TODO need 2 invert mask ?
    mask = np.ma.masked_array(segmentation, np.in1d(segmentation, ignore_segs)).mask
    segmentation[mask] = 0
    segmentation, _ = vigra.analysis.watershedsNew(hmap, seeds=segmentation)
    vigra.analysis.relabelConsecutive(segmentation, out=segmentation)
    return segmentation


# TODO
def merge_fully_enclosed(segmentation, n_threads=8):
    rag = nrag.gridRag(segmentation, numberOfThreads=n_threads)


def merge_mitos(segmentation, mito_map):
    pass
