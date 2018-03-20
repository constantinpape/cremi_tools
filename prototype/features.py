import numpy as np
import vigra
import nifty.graph.rag as nrag
import cremi_tools.segmentation as cseg
from cremi_tools.viewer.volumina import view
from cremi_tools.visualisation import visualize_probabilities


def all_features(affs, seg, offsets):
    rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)
    lifted_uvs, local_features, lifted_features = nrag.computeFeaturesAndNhFromAffinities(rag,
                                                                                          affs,
                                                                                          offsets)
    # return rag, lifted_uvs, local_features, lifted_features
    return local_features[:, 0]


def local_features(affs, seg, offsets):
    feats = cseg.MeanAffinitiyMapFeatures(offsets)
    rag, probs, _, _ = feats(affs, seg)
    return probs


# TODO check random forest features
def nearest_features(affs, seg):
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    feats = cseg.MeanAffinitiyMapFeatures(offsets)
    affs_ = affs[:3]
    rag, probs, _, _ = feats(affs_, seg)
    return probs


if __name__ == '__main__':
    # aff_path = '/home/papec/Work/neurodata_hdd/cremi/sampleB+_affs_cut.h5'
    aff_path = '/home/papec/mnt/papec/sampleB+_affs_cut.h5'
    affs = 1. - vigra.readHDF5(aff_path, 'data')
    lrws = cseg.LRAffinityWatershed(threshold_cc=0.1, threshold_dt=0.2, sigma_seeds=2.)
    ws, _ = lrws(affs)

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]

    feats1 = local_features(affs, ws, offsets)
    feats2 = nearest_features(affs, ws)

    rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1)
    _, attractive1, repulsive1 = visualize_probabilities(rag, feats1, edge_direction=0)
    _, attractive2, repulsive2 = visualize_probabilities(rag, feats2, edge_direction=0)

    # raw_path = '/home/papec/Work/neurodata_hdd/cremi/sampleB+_raw_cut.h5'
    raw_path = '/home/papec/mnt/papec/sampleB+_raw_cut.h5'
    raw = vigra.readHDF5(raw_path, 'data')
    view([raw, ws, attractive1, repulsive1, attractive2, repulsive2],
         ['raw', 'ws', 'attractive_full', 'repulsive_full', 'attractive_local', 'repulsive_local'],
         ['Grayscale', 'RandomColors', 'Green', 'Red', 'Green', 'Red'])
