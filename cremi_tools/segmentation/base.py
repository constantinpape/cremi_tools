from functools import partial
import numpy as np
import nifty
import nifty.graph.rag as nrag

# TODO make number of threads parameter where relevant


class Oversegmenter(object):
    # input needs to be inverted for affinity predictions
    def __init__(self, invert_input=False, return_seeds=False):
        self.invert_input = invert_input
        self.return_seeds = return_seeds

    def __call__(self, input_, mask=None, **kwargs):
        _impl = self._oversegmentation_impl if mask is None else \
            partial(self._oversegmentation_impl_masked, mask=mask)
        if self.invert_input:
            return _impl(1. - input_, **kwargs)
        else:
            return _impl(input_, **kwargs)


class ProblemExtractor(object):

    @staticmethod
    def compute_node_and_edge_sizes(fragments, rag):
        _, node_sizes = np.unique(fragments, return_counts=True)
        edge_sizes = nrag.accumulateEdgeMeanAndLength(rag, np.zeros(rag.shape,
                                                                    dtype='float32'))[:, 1]
        return node_sizes, edge_sizes

    def __call__(self, input_, fragments, **kwargs):
        self.rag = nrag.gridRag(fragments, numberOfLabels=int(fragments.max() + 1))
        probs = self._compute_edge_probabilities(input_, fragments, **kwargs)
        node_sizes, edge_sizes = self.compute_node_and_edge_sizes(fragments, self.rag)
        return self.rag, probs, node_sizes, edge_sizes


class Segmenter(object):
    def __call__(self, graph, costs, node_sizes=None, edge_sizes=None, **kwargs):
        return self._segmentation_impl(graph, costs,
                                       node_sizes=node_sizes, edge_sizes=edge_sizes, **kwargs)

    # dummy implementation
    # subclasses can over-ride
    def probabilities_to_costs(self, probabilities, edge_sizes=None):
        return probabilities


class SegmentationPipeline(object):
    def __init__(self, oversegmenter, extractor, segmenter):
        assert isinstance(oversegmenter, (Oversegmenter, SegmentationPipeline))
        self.oversegmenter = oversegmenter
        assert isinstance(extractor, ProblemExtractor)
        self.extractor = ProblemExtractor
        assert isinstance(segmenter, Segmenter)
        self.segementer = segmenter

    def __call__(self, input_):
        fragments = self.oversegmenter(input_)
        rag, probs, node_sizes, edge_sizes = self.extractor(input_, fragments)
        graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        graph.insertEdges(rag.uvIds())
        costs = self.segmener.probabilities_to_costs(probs, edge_sizes)
        node_labels = self.segmenter(graph, costs, node_sizes, edge_sizes)
        return nrag.projectScalarNodeDataToPixels(rag, node_labels)
