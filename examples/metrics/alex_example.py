from __future__ import print_function
import os
import sys
import time
import numpy as np
import vigra

sys.path.append('../..')
from cremi_tools.metrics import compute_vi_metrics_per_object_improvement


def run_example():

    data_path="/mnt/localdata1/amatskev/debugging/vi_scores/"

    result_files = ['result.h5', 'result_resolved_local.h5']
    datasets=["splA_z0","splA_z1","splB_z0","splB_z1","splC_z0","splC_z1"]
    gt_paths = [data_path + '{}/gt.h5'.format(dataset) for dataset in datasets]

    path_files = [data_path + '{}/false_merge_probs_{}.npy'.format(dataset,dataset) for dataset in datasets]
    path_data_files = [data_path + '{}/paths_to_objs.npy'.format(dataset) for dataset in datasets]

    for ii, dataset in enumerate(datasets):
        key = 'z/{}/data'.format(dataset[-1])

        # load segmentations
        # we need to relabel
        ref_seg, _, mapping = vigra.analysis.relabelConsecutive(
            vigra.readHDF5(data_path + "{}/".format(dataset) + result_files[0], key), start_label=0, keep_zeros=False)
        res_seg = vigra.analysis.relabelConsecutive(
            vigra.readHDF5(data_path + "{}/".format(dataset) + result_files[1], key), start_label=0, keep_zeros=False)[0]

        # load ground trtuh
        gt = vigra.analysis.relabelConsecutive(vigra.readHDF5(gt_paths[ii], 'data'),
                                               start_label=0, keep_zeros=False)[0]

        # Load paths
        paths_to_objs = np.load(path_data_files[ii])
        false_merge_probs = np.load(path_files[ii])

        predicted = false_merge_probs >= 0.3
        object_ids = np.unique(np.array(paths_to_objs)[predicted])
        object_ids = np.array([mapping[obj_id] for obj_id in object_ids], dtype='uint32')

        # TODO use size threshold for gt overlaps
        print("Start computation for seg objs %i / %i" % (len(object_ids), len(np.unique(ref_seg))))
        t_scores = time.time()
        scores_ref, scores_res = compute_vi_metrics_per_object_improvement(ref_seg, res_seg, gt,
                                                                           object_ids=object_ids,
                                                                           overlap_threshold=100)

        print("Compute scores took %f s" % (time.time() - t_scores,))

        np.save(data_path + "{}/scores_ref.npy".format(dataset),scores_ref)
        np.save(data_path + "{}/scores_res.npy".format(dataset),scores_res)

        print(scores_ref)
        print(scores_res)


if __name__ == '__main__':
    run_example()

