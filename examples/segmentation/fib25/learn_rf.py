import sys
import pickle
import os
import vigra
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import nifty.graph.rag as nrag
sys.path.append('../../..')


def learn_rf():
    import cremi_tools.segmentation as cseg
    raw_path = '/home/constantin/Work/neurodata_hdd/FIB25/fib_25_blocks/raw_train_normalized.h5'
    pmap_path = '/home/constantin/Work/neurodata_hdd/FIB25/fib_25_blocks/pmaps/probabilities_train.h5'
    assert os.path.exists(pmap_path), pmap_path
    ws_path = '/home/constantin/Work/neurodata_hdd/FIB25/fib_25_blocks/overseg_train.h5'
    assert os.path.exists(ws_path), ws_path

    # load pmap and watersheds
    raw = vigra.readHDF5(raw_path, 'data').astype('float32')
    pmap = vigra.readHDF5(pmap_path, 'data')
    ws = vigra.readHDF5(ws_path, 'data').astype('uint64')
    assert ws.shape == pmap.shape

    # feature extractor and multicut
    rag = nrag.gridRag(ws, numberOfLabels=int(ws.max() + 1))
    # feature extractor and multicut
    feature_extractor = cseg.FeatureExtractor(True)
    features = np.concatenate([feature_extractor.boundary_map_features(rag, raw),
                               feature_extractor.boundary_map_features(rag, pmap),
                               feature_extractor.region_features(rag, raw, ws)], axis=1)

    gt_path = '/home/constantin/Work/neurodata_hdd/FIB25/fib_25_blocks/gt_train.h5'
    gt = vigra.readHDF5(gt_path, 'data')
    node_labels = nrag.gridRagAccumulateLabels(rag, gt)
    uv_ids = rag.uvIds()
    labels = node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]
    assert len(labels) == len(features), "%i, %i" % (len(labels), len(features))

    print("learning rf from features", features.shape)
    rf = RandomForestClassifier(n_jobs=40, n_estimators=500)
    rf.fit(features, labels)
    with open('./rf.pkl', 'wb') as f:
        pickle.dump(rf, f)


if __name__ == '__main__':
    learn_rf()
