import os
import sys
from collections import OrderedDict
import unittest
import z5py
import numpy as np
sys.path.append('../..')


class TestSkeletonEval(unittest.TestCase):
    # TODO should be setUpClass
    def setUp(self):
        from cremi_tools.skeletons import build_skeleton_metrics
        assert os.path.exists('./test_skeletons.n5')
        assert os.path.exists('./test_ws.n5')
        self.metrics = build_skeleton_metrics('./test_ws.n5/watershed', './test_skeletons.n5')
        self.node_assignment = self.metrics.getNodeAssignments()
        self.skeleton_ids = [int(sk) for sk in os.listdir('./test_skeletons.n5') if sk.isdigit()]
        self.skeleton_ids.sort()

    def compute_split_score(self):
        f_skel = z5py.File('./test_skeletons.n5')
        split_scores = {}
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

    def test_split_score(self):
        split_score1 = self.metrics.computeSplitScores()
        split_score2 = self.compute_split_score()
        self.assertEqual(len(split_score1), len(split_score2))
        self.assertEqual(list(split_score1.keys()), list(split_score2.keys()))
        self.assertEqual(list(split_score1.values()), list(split_score2.values()))

    def compute_runlengths(self, resolution, node_labeling=None):
        assert isinstance(resolution, np.ndarray)
        f_skel = z5py.File('./test_skeletons.n5')
        # we measure runlens for each skeletons and for the fragments in the skelton
        skeleton_runlens = {}
        fragment_runlens = {}

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

    def test_runlength(self):
        resolution = [40., 4., 4.]
        rl1, fl1 = self.metrics.computeSplitRunlengths(resolution)
        rl2, fl2 = self.compute_runlengths(np.array(resolution))
        self.assertEqual(len(rl1), len(rl2))
        self.assertEqual(len(fl1), len(fl2))

        self.assertEqual(list(rl1.keys()), list(rl2.keys()))
        self.assertEqual(list(rl1.values()), list(rl2.values()))

        self.assertEqual(list(fl1.keys()), list(fl2.keys()))
        for key in fl1.keys():
            fll1 = fl1[key]
            fll2 = fl2[key]
            fll2o = OrderedDict(sorted(fll2.items()))
            self.assertEqual(list(fll1.keys()), list(fll2o.keys()))
            self.assertEqual(list(fll1.values()), list(fll2o.values()))

    def compute_explicit_merges(self):
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

    def test_merge_score(self):
        merges1 = self.metrics.computeExplicitMerges()
        merges2 = self.compute_explicit_merges()
        self.assertEqual(len(merges1), len(merges2))
        self.assertEqual(list(merges1.keys()), list(merges2.keys()))


if __name__ == '__main__':
    unittest.main()
