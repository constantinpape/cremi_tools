import os
import pickle
import numpy as np
import z5py
import nifty.graph.rag as nrag
from sklearn.ensemble import RandomForestClassifier


def extract_features_and_labels(sample):
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    path = '/home/papec/mnt/papec/Work/neurodata_hdd/cremi_warped/sample%s_train.n5' % sample
    f = z5py.File(path)
    ws = f['segmentations/watershed'][:]
    rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1)

    affs = 1. - f['predictions/full_affs'][:]
    lifted_uvs, local_features, lifted_features = nrag.computeFeaturesAndNhFromAffinities(rag,
                                                                                          affs,
                                                                                          offsets)

    gt = f['segmentations/groundtruth'][:]
    node_labels = nrag.gridRagAccumulateLabels(rag, gt)

    uv_ids = rag.uvIds()

    local_valid_edges = (node_labels[uv_ids] != 0).all(axis=1)
    local_labels = (node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]).astype('uint8')
    assert len(local_features) == len(local_labels), "%i, %i" % (len(local_features), len(local_labels))

    lifted_valid_edges = (node_labels[lifted_uvs] != 0).all(axis=1)
    lifted_labels = (node_labels[lifted_uvs[:, 0]] != node_labels[lifted_uvs[:, 1]]).astype('uint8')
    assert len(lifted_features) == len(lifted_labels), "%i, %i" % (len(lifted_features), len(lifted_labels))

    print("Number of valid local edges", np.sum(local_valid_edges), local_valid_edges.size)
    print("Number of valid lifted edges", np.sum(lifted_valid_edges), lifted_valid_edges.size)

    local_labels = local_labels[local_valid_edges]
    local_features = local_features[local_valid_edges]
    assert len(local_features) == len(local_labels), "%i, %i" % (len(local_features), len(local_labels))

    lifted_labels = lifted_labels[lifted_valid_edges]
    lifted_features = lifted_features[lifted_valid_edges]
    assert len(lifted_features) == len(lifted_labels), "%i, %i" % (len(lifted_features), len(lifted_labels))

    return local_labels, local_features, lifted_labels, lifted_features


if __name__ == '__main__':
    samples = ('A', 'B', 'C')
    # samples = ('A',)

    local_features = []
    local_labels = []

    lifted_features = []
    lifted_labels = []

    for sample in samples:
        print("Computing features for sample", sample)
        local_labs, local_feats, lifted_labs, lifted_feats = extract_features_and_labels(sample)

        local_features.append(local_feats)
        local_labels.append(local_labs)

        lifted_features.append(lifted_feats)
        lifted_labels.append(lifted_labs)

    local_features = np.concatenate(local_features, axis=0)
    local_labels = np.concatenate(local_labels, axis=0)
    assert len(local_features) == len(local_labels), "%i, %i" % (len(local_features), len(local_labels))

    print("Local feature shape", local_features.shape)
    print("Local labels shape", local_labels.shape)

    lifted_features = np.concatenate(lifted_features, axis=0)
    lifted_labels = np.concatenate(lifted_labels, axis=0)
    assert len(lifted_features) == len(lifted_labels)

    print("lifted feature shape", lifted_features.shape)
    print("lifted labels shape", lifted_labels.shape)

    # train random forests
    rf1 = RandomForestClassifier(n_jobs=8, n_estimators=100)
    rf1.fit(local_features, local_labels)

    rf_folder = '/home/papec/mnt/papec/Work/neurodata_hdd/cremi_warped/random_forests'
    with open(os.path.join(rf_folder, 'rf_ABC_local_affinity_feats.pkl'), 'wb') as f:
        pickle.dump(rf1, f)

    rf2 = RandomForestClassifier(n_jobs=8, n_estimators=100)
    rf2.fit(lifted_features, lifted_labels)

    with open(os.path.join(rf_folder, 'rf_ABC_lifted_affinity_feats.pkl'), 'wb') as f:
        pickle.dump(rf2, f)

    features = np.concatenate([local_features, lifted_features], axis=0)
    labels = np.concatenate([local_labels, lifted_labels], axis=0)
    rf3 = RandomForestClassifier(n_jobs=8, n_estimators=100)
    rf3.fit(features, labels)

    with open(os.path.join(rf_folder, 'rf_ABC_all_affinity_feats.pkl'), 'wb') as f:
        pickle.dump(rf3, f)
