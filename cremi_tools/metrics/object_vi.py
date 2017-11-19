from __future__ import print_function
import os
from itertools import chain
import numpy as np
from concurrent import futures

# from .vi_metrics import compute_vi_metrics

# TODO generalize this with vi_metrics

# TODO this should be documentation...
# per object VI-scores following
# https://arxiv.org/pdf/1708.02599.pdf
# the object split (vi_split) and merge (vi_merge) scores are given by
# vi_split = - sum_j ( r_ij / p_i * log(r_ij / p_i) )
# vi_merge = - sum_j ( r_ij / p_i * log(r_ij / q_j) )
# with
# r_ij = number of pixels in common between gt-segment i and proposed segment j
# p_i = sum_j r_ij  # number of pixels in gt-segment i
# q_j = sum_i r_iJ  # number of pixels in proposed segmnet j
def compute_vi_metrics(sub_segmentation, sub_groundtruth, object_counts):

    # get the obj ids and counts for objects in the sub segmentation
    seg_objs, counts_in_sub = np.unique(sub_segmentation, return_counts=True)
    counts_in_full = object_counts[seg_objs]

    # compute r_ij, q_j and p_i
    r_ij = np.array(counts_in_sub)
    q_j = np.array(counts_in_full,dtype="float32")
    p_i = float(sub_groundtruth.size)

    # compute the first factor in term
    r_div_p = np.divide(r_ij, p_i)
    r_div_q = np.divide(r_ij, q_j)

    # compute the second factor in term
    log_r_q = np.log(r_div_q)
    log_r_p = np.log(r_div_p)

    # compute sum
    merge_score = -np.sum(np.multiply(r_div_p, log_r_q))
    split_score = -np.sum(np.multiply(r_div_p, log_r_p))

    return merge_score, split_score


def compute_vi_metrics_per_object(segmentation,
                                  groundtruth,
                                  object_ids=None,
                                  overlap_threshold=0,
                                  n_workers=8):
    vi_obj_scores = []

    # we need to precompute the object counts for both segmentations
    all_ids, counts = np.unique(segmentation, return_counts=True)

    # if the list of objects was not specified, we calculate the core for all
    # objects
    if object_ids is None:
        object_ids = all_ids

    def scores_for_object(obj_id):
        # find the ground-truth objects that have overlap with the reference
        # segmentation object
        print("Computing object vi for obj id", obj_id)
        gt_objects, obj_counts = np.unique(groundtruth[segmentation == obj_id], return_counts=True)
        # apply size threshold if specified
        if overlap_threshold:
            gt_objects = gt_objects[obj_counts > overlap_threshold]

        results = []
        for gt_obj in gt_objects:
            gt_obj_mask = groundtruth == gt_obj
            gt_obj_seg = groundtruth[gt_obj_mask]
            results.append(compute_vi_metrics(segmentation[gt_obj_mask], gt_obj_seg, counts))

        return results

    with futures.ThreadPoolExecutor(max_workers=n_workers) as tp:
        tasks = [tp.submit(scores_for_object, obj_id) for obj_id in object_ids]
        # need to get the result and flatten it
        results = list(chain.from_iterable([t.result() for t in tasks]))
        assert len(results[0] == len(object_ids))
        assert len(results[1] == len(object_ids))

    obj_scores = {obj_id: scores for obj_id, scores in zip(object_ids, results)}
    return obj_scores


def compute_vi_metrics_per_object_improvement(reference_segmentation,
                                              resolved_segmentation,
                                              groundtruth,
                                              object_ids=None,
                                              overlap_threshold=0,
                                              n_workers=8):
    # TODO update doc string
    """Return the contingency table for all regions in matched segmentations.

        Parameters
        ----------
        init_seg :              Initial segmentation
        res_seg :               Resolved segmentation
        gt :                    Groundtruth segmentation
        ids_of_objects_changed :Ids of objects changed

        Returns
        -------
        vi_obj_scores_summed_initial:   np.array with total sum of [merge,split] for every id changed in init_seg
        vi_obj_scores_summed_resolved:  np.array with total sum of [merge,split] for every id changed in res_seg
        ids_of_objects_changed:         list of all ids where res_seg has been resolved

        """

    vi_obj_scores_reference = []
    vi_obj_scores_resolved = []

    # we need to precompute the object counts for both segmentations
    reference_ids, reference_counts = np.unique(reference_segmentation, return_counts=True)
    _, resolved_counts = np.unique(resolved_segmentation, return_counts=True)

    # if the list of objects was not specified, we calculate the core for all
    # objects
    if object_ids is None:
        object_ids = reference_ids

    n_objects = len(object_ids)

    def scores_for_object(obj_id, index):
        # find the ground-truth objects that have overlap with the reference
        # segmentation object
        print("Computing object vi for obj id %i: %i / %i" % (obj_id, index, n_objects))
        gt_objects, obj_counts = np.unique(groundtruth[reference_segmentation == obj_id],
                                           return_counts=True)
        # apply size threshold if specified
        if overlap_threshold:
            gt_objects = gt_objects[obj_counts > overlap_threshold]

        reference_results = []
        resolved_results = []
        for gt_obj in gt_objects:
            gt_obj_mask = groundtruth == gt_obj
            gt_obj_seg = groundtruth[gt_obj_mask]
            reference_results.append(compute_vi_metrics(reference_segmentation[gt_obj_mask], gt_obj_seg, reference_counts))
            resolved_results.append(compute_vi_metrics(resolved_segmentation[gt_obj_mask], gt_obj_seg, resolved_counts))

        return reference_results, resolved_results

    with futures.ThreadPoolExecutor(max_workers=n_workers) as tp:
        tasks = [tp.submit(scores_for_object, obj_id, ii) for ii, obj_id in enumerate(object_ids)]
        results = np.array([t.result() for t in tasks])
        assert len(results) == len(object_ids), "%i, %i" % (len(results), len(object_ids))

    obj_scores_reference = {obj_id: obj_scores for obj_id, obj_scores in zip(object_ids, results[:, 0])}
    obj_scores_resolved = {obj_id: obj_scores for obj_id, obj_scores in zip(object_ids, results[:, 1])}
    return obj_scores_reference, obj_scores_resolved
