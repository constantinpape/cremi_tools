import sys
import unittest
# import os
# import z5py
import numpy as np
sys.path.append('../..')


class TestTreeSmoothing(unittest.TestCase):
    def setUp(self):
        # toy tree data:
        # 1 - 2, 2 - 3, 3 - 4, 3 - 5, 4 - 6, 6 - 7, 5 - 8, 8 - 9, 9 - 10
        self.edge_list0 = np.array([[1, 2],
                                    [2, 3],
                                    [3, 4],
                                    [3, 5],
                                    [4, 6],
                                    [6, 7],
                                    [5, 8],
                                    [8, 9],
                                    [9, 10]], dtype='uint32')
        values = np.unique(self.edge_list0)
        # for the toy data, each node has his own id as value (list)
        self.values0 = {v: [v] for v in values}

    def test_sliding_window_neighbors(self):
        from cremi_tools.skeletons.tree_smoothing import neighbors_in_window, build_skeleton_tree
        edges = self.edge_list0
        nodes = np.unique(edges)
        tree = build_skeleton_tree(edges)

        max_depth = 1  # depth 1 corresponds to sliding window size of 3 (including the node itself)
        expected_nbrs = {1: [2], 2: [1, 3], 3: [],
                         4: [3, 6], 5: [3, 8], 6: [4, 7],
                         7: [6], 8: [5, 9], 9: [8, 10], 10: [9]}
        for node in nodes:
            nbrs, dpts = neighbors_in_window(tree, node, max_depth)
            nbrs.sort()
            expected = expected_nbrs[node]
            self.assertEqual(nbrs, expected)
            self.assertEqual(dpts, [1] * len(nbrs))

        max_depth = 2  # depth 2 corresponds to sliding window size of 5 (including the node itself)
        expected_nbrs = {1: [2, 3], 2: [1, 3], 3: [],
                         4: [3, 6, 7], 5: [3, 8, 9], 6: [3, 4, 7],
                         7: [4, 6], 8: [3, 5, 9, 10], 9: [5, 8, 10], 10: [8, 9]}
        for node in nodes:
            nbrs, dpts = neighbors_in_window(tree, node, max_depth)
            nbrs.sort()
            expected = expected_nbrs[node]
            self.assertEqual(nbrs, expected)
            self.assertEqual(len(dpts), len(nbrs))

    def test_bfs_neighbors(self):
        from cremi_tools.skeletons.tree_smoothing import neighbors_bfs, build_skeleton_tree
        edges = self.edge_list0
        nodes = np.unique(edges)
        tree = build_skeleton_tree(edges)

        max_depth = 1
        expected_nbrs = {1: [2], 2: [1, 3], 3: [2, 4, 5],
                         4: [3, 6], 5: [3, 8], 6: [4, 7],
                         7: [6], 8: [5, 9], 9: [8, 10], 10: [9]}
        for node in nodes:
            nbrs, dpts = neighbors_bfs(tree, node, max_depth)
            nbrs.sort()
            expected = expected_nbrs[node]
            self.assertEqual(nbrs, expected)
            self.assertEqual(dpts, [1] * len(nbrs))

        max_depth = 2
        expected_nbrs = {1: [2, 3], 2: [1, 3, 4, 5], 3: [1, 2, 4, 5, 6, 8],
                         4: [2, 3, 5, 6, 7], 5: [2, 3, 4, 8, 9], 6: [3, 4, 7],
                         7: [4, 6], 8: [3, 5, 9, 10], 9: [5, 8, 10], 10: [8, 9]}
        for node in nodes:
            nbrs, dpts = neighbors_bfs(tree, node, max_depth)
            nbrs.sort()
            expected = expected_nbrs[node]
            self.assertEqual(nbrs, expected)
            self.assertEqual(len(dpts), len(nbrs))

    def test_smoothing_window(self):
        from cremi_tools.skeletons import smooth_sliding_window
        nodes = np.unique(self.edge_list0)

        max_depth = 1  # depth 1 corresponds to sliding window size of 3 (including the node itself)
        smoothed_values = smooth_sliding_window(self.values0, self.edge_list0, max_depth)
        expected_values = {1: [1, 2], 2: [1, 2, 3], 3: [3],
                           4: [3, 4, 6], 5: [3, 5, 8], 6: [4, 6, 7],
                           7: [6, 7], 8: [5, 8, 9], 9: [8, 9, 10], 10: [9, 10]}
        for node in nodes:
            vals = smoothed_values[node]
            vals.sort()
            expected = expected_values[node]
            self.assertEqual(vals, expected)

        max_depth = 2  # depth 2 corresponds to sliding window size of 5 (including the node itself)
        smoothed_values = smooth_sliding_window(self.values0, self.edge_list0, max_depth)
        expected_values = {1: [1, 2, 3], 2: [1, 2, 3], 3: [3],
                           4: [3, 4, 6, 7], 5: [3, 5, 8, 9], 6: [3, 4, 6, 7],
                           7: [4, 6, 7], 8: [3, 5, 8, 9, 10], 9: [5, 8, 9, 10], 10: [8, 9, 10]}
        for node in nodes:
            vals = smoothed_values[node]
            vals.sort()
            expected = expected_values[node]
            self.assertEqual(vals, expected)

    def test_smoothing_bfs(self):
        from cremi_tools.skeletons import smooth_bfs
        nodes = np.unique(self.edge_list0)

        max_depth = 1
        smoothed_values = smooth_bfs(self.values0, self.edge_list0, max_depth)
        expected_values = {1: [1, 2], 2: [1, 2, 3], 3: [2, 3, 4, 5],
                           4: [3, 4, 6], 5: [3, 5, 8], 6: [4, 6, 7],
                           7: [6, 7], 8: [5, 8, 9], 9: [8, 9, 10], 10: [9, 10]}
        for node in nodes:
            vals = smoothed_values[node]
            vals.sort()
            expected = expected_values[node]
            self.assertEqual(vals, expected)

        max_depth = 2
        smoothed_values = smooth_bfs(self.values0, self.edge_list0, max_depth)
        expected_values = {1: [1, 2, 3], 2: [1, 2, 3, 4, 5], 3: [1, 2, 3, 4, 5, 6, 8],
                           4: [2, 3, 4, 5, 6, 7], 5: [2, 3, 4, 5, 8, 9], 6: [3, 4, 6, 7],
                           7: [4, 6, 7], 8: [3, 5, 8, 9, 10], 9: [5, 8, 9, 10], 10: [8, 9, 10]}
        for node in nodes:
            vals = smoothed_values[node]
            vals.sort()
            expected = expected_values[node]
            self.assertEqual(vals, expected)


if __name__ == '__main__':
    unittest.main()
