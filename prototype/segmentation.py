import os
import time
import pickle
from concurrent import futures

import numpy as np
import vigra
import nifty
import nifty.graph.rag as nrag
import cremi_tools.segmentation as cseg
from cremi_tools.viewer.volumina import view


# import nifty.mws as nmws
# import nifty.graph.rag as nrag
# import z5py


def full_features(rag, affs, offsets):
    lifted_uvs, local_features, lifted_features = nrag.computeFeaturesAndNhFromAffinities(rag,
                                                                                          affs,
                                                                                          offsets)
    return lifted_uvs, np.nan_to_num(local_features), np.nan_to_num(lifted_features)


def nearest_features(rag, affs):
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    affs_ = affs[:3]
    probs = nrag.accumulateAffinityStandartFeatures(rag, affs_, offsets)[:, 0]
    return probs


def thresholding_watersheds_2d(affs, n_threads=8, threshold=0.05, sigma=2.):
    hmap = np.mean(affs[1:3], axis=0) + np.mean(affs[4:6], axis=0)
    hmap /= 2

    seg = np.zeros_like(hmap, dtype='uint32')

    def run_ws_z(z):
        hmapz = hmap[z]
        if sigma > 0.:
            hmapz = np.clip(vigra.filters.gaussianSmoothing(hmapz, sigma=sigma), 0, 1)
        seeds = hmapz <= 0.05
        seeds = vigra.analysis.labelImageWithBackground(seeds.view('uint8'))
        seg_z, max_z = vigra.analysis.watershedsNew(hmapz, seeds=seeds)
        seg[z] = seg_z
        return max_z

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(run_ws_z, z) for z in range(seg.shape[0])]
        offsets = np.array([t.result() for t in tasks], dtype='uint32')

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    seg += offsets[:, None, None]

    return seg


def thresholding_watersheds_3d(affs, threshold=0.05, sigma=2.):
    seed_map = np.mean(affs[:3], axis=0) + np.mean(affs[4:6], axis=0)
    seed_map /= 2

    hmap = np.mean(affs[1:3], axis=0)

    seg = np.zeros_like(hmap, dtype='uint32')

    if sigma > 0.:
        seed_map = np.clip(vigra.filters.gaussianSmoothing(seed_map, sigma=sigma), 0, 1)
    seeds = seed_map <= 0.05

    seeds = vigra.analysis.labelVolumeWithBackground(seeds.view('uint8'))
    seg, maxx = vigra.analysis.watershedsNew(hmap, seeds=seeds)
    return seg


def compare_ws():
    aff_path = '/home/papec/mnt/papec/sampleB+_affs_cut.h5'
    affs = 1. - vigra.readHDF5(aff_path, 'data')

    ws0 = thresholding_watersheds_2d(affs)
    ws1 = thresholding_watersheds_2d(affs, sigma=0)

    ws2 = thresholding_watersheds_3d(affs)
    ws3 = thresholding_watersheds_3d(affs, sigma=0)

    lr = cseg.LRAffinityWatershed(threshold_cc=0.1, threshold_dt=0.2, sigma_seeds=2.)
    ws4, _ = lr(affs)

    raw_path = '/home/papec/mnt/papec/sampleB+_raw_cut.h5'
    raw = vigra.readHDF5(raw_path, 'data')
    view([raw, ws0, ws1, ws2, ws3, ws4], ['raw', 'ws-2d', 'ws-2d-1', 'ws-3d', 'ws-3d-1', 'lr-ws'])


def mc(rag, probs):
    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())
    costs = mc.probabilities_to_costs(probs)
    node_labels = mc(graph, costs)
    return nrag.projectScalarNodeDataToPixels(rag, node_labels)


def get_lifted_problem(affs, seg, offsets):
    rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)
    lifted_uvs, local_features, lifted_features = nrag.computeFeaturesAndNhFromAffinities(rag,
                                                                                          affs,
                                                                                          offsets)
    return rag, lifted_uvs, local_features, lifted_features


def lmc(rag, lifted_uvs, local_features, lifted_features):
    lmc = cseg.LiftedMulticut('kernighan-lin', weight_edges=False)
    local_costs = lmc.probabilities_to_costs(local_features)
    lifted_costs = lmc.probabilities_to_costs(lifted_features)
    node_labels = lmc(rag.uvIds(), lifted_uvs, local_costs, lifted_costs)
    return nrag.projectScalarNodeDataToPixels(rag, node_labels)


def mws_clustering(rag, lifted_uvs, local_features, lifted_features):
    import nifty.mws as nmws
    node_labels = nmws.computeMwsClustering(rag.numberOfNodes, rag.uvIds(), lifted_uvs,
                                            local_features[:, 0], 1. - lifted_features[:, 0])
    return nrag.projectScalarNodeDataToPixels(rag, node_labels)


# Timing
# Building grid graph...
# ... in 0.000018 s
# Extracting problem...
# ... in 0.674187 s
# Computing mws...
# ... in 56.531404 s
def mws_segmentation(affs, offsets, strides=None):
    import nifty.mws as nmws
    shape = affs.shape[1:]

    print("Building grid graph...")
    t0 = time.time()
    graph = nifty.graph.undirectedGridGraph(shape)
    print("... in %f s" % (time.time() - t0,))

    print("Extracting problem...")
    t0 = time.time()
    if strides is None:
        n_lifted, local_probs, lifted_uvs, lifted_probs = graph.liftedProblemFromLongRangeAffinities(affs,
                                                                                                     offsets)
    else:
        assert len(strides) == 3
        n_lifted, local_probs, lifted_uvs, lifted_probs = graph.liftedProblemFromLongRangeAffinitiesWithStrides(affs,
                                                                                                                offsets,
                                                                                                                strides)
    # FIXME this should be done in c++, but resizing does weird things
    lifted_uvs = lifted_uvs[:n_lifted]
    lifted_probs = lifted_probs[:n_lifted]
    assert lifted_uvs.shape[1] == 2
    assert len(lifted_uvs) == len(lifted_probs)
    print("... in %f s" % (time.time() - t0,))

    print("Computing mws...")
    t0 = time.time()
    node_labels = nmws.computeMwsClustering(graph.numberOfNodes, graph.uvIds(), lifted_uvs,
                                            local_probs, 1. - lifted_probs)
    print("... in %f s" % (time.time() - t0,))
    return node_labels.reshape(shape)


# Computing mws ...
# ... in 207.485458 s
def mutex_segmentation(affs, offsets):
    import constrained_mst as cmst
    mst = cmst.ConstrainedWatershed(np.array(affs.shape[1:]),
                                    np.array(offsets), 3,
                                    np.array([1, 1, 1]))
    affs[3:] *= -1
    affs[3:] += 1
    print("Computing mws ...")
    t0 = time.time()
    sorted_edges = np.argsort(affs.ravel())
    mst.repulsive_ucc_mst_cut(sorted_edges, 0)
    print("... in %f s" % (time.time() - t0,))
    seg = mst.get_flat_label_image().reshape(affs.shape[1:])
    return seg


def compare_all_segmentations():
    aff_path = '/home/papec/mnt/papec/sampleB+_affs_cut.h5'
    print("Loading affinities")
    affs = 1. - vigra.readHDF5(aff_path, 'data')
    print(affs.dtype, affs.min(), affs.max())
    print("Computing watershed")
    lrws = cseg.LRAffinityWatershed(threshold_cc=0.1, threshold_dt=0.2, sigma_seeds=2.)
    ws, n_labels = lrws(affs)
    print("Computing RAG")
    rag = nrag.gridRag(ws, numberOfLabels=n_labels+1)

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]

    print("computing features")
    lifted_uvs, local_features, lifted_features = full_features(rag, affs, offsets)
    # nearest_probs = nearest_features(rag, affs)
    local_probs = local_features[:, 0]
    lifted_probs = lifted_features[:, 0]

    # load random forests
    rf_folder = '/home/papec/mnt/papec/Work/neurodata_hdd/cremi_warped/random_forests'
    with open(os.path.join(rf_folder, 'rf_ABC_local_affinity_feats.pkl'), 'rb') as f:
        rf1 = pickle.load(f)

    rf_local_probs = rf1.predict_proba(local_features)[:, 1]
    with open(os.path.join(rf_folder, 'rf_ABC_lifted_affinity_feats.pkl'), 'rb') as f:
        rf2 = pickle.load(f)
    rf_lifted_probs = rf2.predict_proba(lifted_features)[:, 1]

    print("computing multicuts")
    mc_local = mc(rag, local_probs)
    # mc_nearest = mc(rag, nearest_probs)
    mc_rf = mc(rag, rf_local_probs)

    print("computing lifted multicuts")
    lmc_local = lmc(rag, lifted_uvs, local_probs, lifted_probs)
    # lmc_nearest = lmc(rag, lifted_uvs, nearest_probs, lifted_probs)
    lmc_rf = lmc(rag, lifted_uvs, rf_local_probs, rf_lifted_probs)

    # print("Running MWS clustering")
    # mws_seg = mws_clustering(*lifted_problem)

    raw_path = '/home/papec/mnt/papec/sampleB+_raw_cut.h5'
    raw = vigra.readHDF5(raw_path, 'data')
    view([raw, ws, mc_local, mc_rf, lmc_local, lmc_rf],
         ['raw', 'ws', 'mc-local', 'mc-rf', 'lmc-local', 'lmc-rf'])
    # view([raw, ws, mc_local, mc_rf],
    #      ['raw', 'ws', 'mc-local', 'mc-rf'])


def compare_mws_pixelwise():
    aff_path = '/home/papec/Work/neurodata_hdd/cremi/sampleB+_affs_cut.h5'
    affs = 1. - vigra.readHDF5(aff_path, 'data')
    bb = np.s_[:10, :]
    affs = affs[(slice(None),) + bb]
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]

    strides = [1, 2, 2]
    mws1 = mws_segmentation(affs, offsets, strides)
    # mws2 = mutex_segmentation(affs, offsets)

    raw_path = '/home/papec/Work/neurodata_hdd/cremi/sampleB+_raw_cut.h5'
    raw = vigra.readHDF5(raw_path, 'data')[bb]
    view([raw, mws1])
    # view([raw, mws1, mws2])


if __name__ == '__main__':
    compare_all_segmentations()
