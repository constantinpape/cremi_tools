import numpy as np
import nifty.graph.rag as nrag


class SegmentationComparison(object):
    def __init__(self, base_segmentation, n_threads=8):
        assert isinstance(base_segmentation, np.ndarray)
        self.base_segmentation = base_segmentation
        self.n_threads = n_threads
        self.rag = nrag.gridRag(self.base_segmentation,
                                self.n_threads)
        self.uv_ids = self.rag.uvIds()
        self.volume_builder = nrag.ragCoordinates(self.rag, self.n_threads)

    def active_edges(self, seg):
        nodes = nrag.gridRagAccumulateLabels(self.rag, seg)
        return nodes[self.uv_ids[:, 0]] != nodes[self.uv_ids[:, 1]]

    def active_edge_volume(self, seg):
        edges = self.active_edges(seg)
        return self.volume_builder.edgesToVolume(edges,
                                                 edgeDirection=0,
                                                 numberOfThreads=self.n_threads)

    def edge_difference(self, seg_a, seg_b):
        return self.active_edges(seg_a) != self.active_edges(seg_b)

    def edge_difference_volume(self, seg_a, seg_b):
        edge_diff = self.edge_difference(seg_a, seg_b).astype('uint8')
        return self.volume_builder.edgesToVolume(edge_diff,
                                                 edgeDirection=0,
                                                 numberOfThreads=self.n_threads)
