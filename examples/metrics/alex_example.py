from __future__ import print_function
import os
import sys
import time
import numpy as np
import vigra

sys.path.append('../..')
from cremi_tools.metrics import compute_vi_metrics_per_object_improvement


def run_example():
    result_files = ['result.h5', 'result_resolved_local.h5']

    halves = [0, 1]
    gt_paths = ['./splA_z%i/gt.h5' % half for half in halves]

    path_files = ['./false_merge_probs_splA_z%i.npy' % half for half in halves]
    path_data_files = ['./splA_z%i/paths_to_objs.npy' % half for half in halves]

    for ii, half in enumerate(halves):
        key = 'z/%i/data' % half

        # load segmentations
        # we need to relabel
        ref_seg, _, mapping = vigra.analysis.relabelConsecutive(
            vigra.readHDF5(result_files[0], key), start_label=0, keep_zeros=False)
        res_seg = vigra.analysis.relabelConsecutive(
            vigra.readHDF5(result_files[1], key), start_label=0, keep_zeros=False)[0]

        # load ground trtuh
        gt = vigra.analysis.relabelConsecutive(vigra.readHDF5(gt_paths[ii], 'data'),
                                               start_label=0, keep_zeros=False)[0]

        # Load paths
        paths_to_objs = np.load(path_data_files[ii])
        false_merge_probs = np.load(path_files[ii])

        predicted = false_merge_probs >= 0.3
        object_ids = np.unique(np.array(paths_to_objs)[predicted])
        object_ids = np.array([mapping[obj_id] for obj_id in object_ids], dtype='uint32')[:2]

        # TODO use size threshold for gt overlaps
        print("Start computation for seg objs %i / %i" % (len(object_ids), len(np.unique(ref_seg))))
        t_scores = time.time()
        scores_ref, scores_res = compute_vi_metrics_per_object_improvement(ref_seg, res_seg, gt,
                                                                           object_ids=object_ids,
                                                                           overlap_threshold=100)
        print("Compute scores took %f s" % (time.time() - t_scores,))
        print(scores_ref)
        print(scores_res)


if __name__ == '__main__':
    run_example()
