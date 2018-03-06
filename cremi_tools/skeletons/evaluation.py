import os
import numpy as np
import z5py
import nifty.skeletons as nskel


# TODO enable serialization of node_assignment
class SkeletonMetrics(object):
    def __init__(self, label_file, skeleton_file, n_threads):
        # validate the input paths
        assert os.path.exists(label_file), label_file
        self.label_file = label_file
        assert os.path.exists(skeleton_file), skeleton_file
        self.skeleton_file = skeleton_file
        self.n_threads = n_threads

        # compute the node assignment
        self.skeleton_ids = os.listdir(self.skeleton_file)
        self.skeleton_ids = [int(sk) for sk in self.skeleton_ids if sk.isdigit()]
        self.skeleton_ids.sort()
        # TODO just prelim
        skel_metrics = nskel.SkeletonMetrics(self.label_file, self.skeleton_file,
                                             self.skeleton_ids, self.n_threads)
        self.node_assignment = skel_metrics.getNodeAssignments()
        assert len(self.node_assignment) == len(self.skeleton_ids)

    # TODO enable split-score with node relabeling (e.g. for agglomeration result from multicut)
    # TODO debug mode to get edges that contain the false split
    def compute_split_score(self, node_labeling=None):
        f_skel = z5py.File(self.skeleton_file)
        split_scores = {}
        # TODO we could vectorize here
        for skel_id in self.skeleton_ids:
            skel_assignment = self.node_assignment[skel_id]
            edges = f_skel['%i/edges' % skel_id][:]
            n_edges = len(edges)
            split_score = 0.
            for e in edges:
                node_a, node_b = skel_assignment[e[0]], skel_assignment[e[1]]
                if node_a == node_b:
                    split_score += 1
            split_scores[skel_id] = split_score / n_edges
        return split_scores

    # TODO enable split-score with node relabeling (e.g. for agglomeration result from multicut)
    def compute_split_runlength(self, resolution, node_labeling=None):
        assert isinstance(resolution, np.ndarray)
        f_skel = z5py.File(self.skeleton_file)
        # we measure runlens for each skeletons and for the fragments in the skelton
        skeleton_runlens = {}
        fragment_runlens = {}

        # TODO we could vectorize here
        for skel_id in self.skeleton_ids:

            # load and initialize
            skel_assignment = self.node_assignment[skel_id]
            edges = f_skel['%i/edges' % skel_id][:]
            coords = f_skel['%i/coordinates' % skel_id][:]
            skel_runlen = 0
            fragment_runlen = {}

            # get dictionary from node-id to coordinate
            node_to_coord = {coord[0]: i for i, coord in enumerate(coords)}

            # bring coordinates to proper resolution and slice away the node id
            coords = coords[:, 1:].astype('float64')
            coords *= resolution

            # iterate over the edges
            for e in edges:
                node_a, node_b = skel_assignment[e[0]], skel_assignment[e[1]]
                coord_a, coord_b = coords[node_to_coord[e[0]]], coords[node_to_coord[e[1]]]
                coord_len = np.sqrt(np.sum(np.square(coord_a - coord_b)))
                skel_runlen += coord_len
                # we only count this part of a skeleton as runlen for a fragment if the node assignment is
                # unambiguous
                if node_a == node_b:
                    if node_a in fragment_runlen:
                        fragment_runlen[node_a] += coord_len
                    else:
                        fragment_runlen[node_a] = coord_len
            skeleton_runlens[skel_id] = skel_runlen
            fragment_runlens[skel_id] = fragment_runlen

        return skeleton_runlens, fragment_runlens

    def compute_explicit_merges(self, node_labeling=None):
        skeletons_with_merge = {}
        # find the unique labels for each skeleton
        labels_per_skeleton = {skel_id: np.unique(list(labels.values()))
                               for skel_id, labels in self.node_assignment.items()}

        # get all segment labels that share ovelap with at least one skeleton node
        labels = np.concatenate([labs for labs in labels_per_skeleton.values()])
        label_len = len(labels)
        labels = np.unique(labels)
        unique_len = len(labels)

        # if the number of labels is equal to the number of unique labels, we have no merges and can return
        if label_len == unique_len:
            return skeletons_with_merge

        for label_id in labels:
            skels_with_label = [skel_id for skel_id, labs in labels_per_skeleton.items()
                                if label_id in labs]
            if len(skels_with_label) > 1:
                for skel_id in skels_with_label:
                    if skel_id in skeletons_with_merge:
                        skeletons_with_merge[skel_id].append(label_id)
                    else:
                        skeletons_with_merge[skel_id] = [label_id]
        return skeletons_with_merge
