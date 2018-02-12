import nifty.graph.rag as nrag
from .base import ProblemExtractor

# Feature extractors yield:
# 0 mean
# 1 variance
# 2 min
# 3 q10
# 4 q25
# 5 q50 / median
# 6 q75
# 7 q90
# 8 max

STAT_TO_INDEX = {'mean': 0, 'min': 2,
                 'median': 5, 'max': 8,
                 'quantile10': 3, 'quantile25': 4,
                 'quantile50': 5, 'quantile75': 6,
                 'quantile90': 7}


class MeanAffinitiyMapFeatures(ProblemExtractor):
    def __init__(self, offsets, statistic='mean'):
        assert isinstance(offsets, list)
        assert all(isinstance(off, list) for off in offsets)
        assert all(len(off) == 3 for off in offsets)
        assert statistic in STAT_TO_INDEX
        self.stat_index = STAT_TO_INDEX[statistic]

    def _compute_edge_probabilities(self, input_, fragments=None):
        assert input_.ndim == 4
        assert input_.shape[0] == len(self.offsets)
        features = nrag.accumulateAffinityStandartFeatures(self.rag, input_, self.offsets)
        return features[:, self.stat_index]


class MeanBoundaryMapFeatures(ProblemExtractor):
    def __init__(self, statistic='mean'):
        assert statistic in STAT_TO_INDEX
        self.stat_index = STAT_TO_INDEX[statistic]

    def _compute_edge_probabilities(self, input_, fragments=None):
        assert input_.ndim == 3
        features = nrag.accumulateEdgeStandartFeatures(self.rag, input_)
        return features[:, self.stat_index]
