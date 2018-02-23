import os
import pickle
import numpy as np
import nifty.graph.rag as nrag

try:
    import fastfiters as filters
except ImportError as e:
    import vigra.filters as filters

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
        self.offsets = offsets

    def _compute_edge_probabilities(self, input_, fragments=None):
        assert input_.ndim == 4
        assert input_.shape[0] == len(self.offsets), "%i, %i" % (input_.shape[0], len(self.offsets))
        features = nrag.accumulateAffinityStandartFeatures(self.rag, input_, self.offsets)
        return features[:, self.stat_index]


class MeanBoundaryMapFeatures(ProblemExtractor):
    def __init__(self, statistic='mean', min_value=0., max_value=1.):
        assert statistic in STAT_TO_INDEX
        self.stat_index = STAT_TO_INDEX[statistic]
        self.min_value = min_value
        self.max_value = max_value

    def _compute_edge_probabilities(self, input_, fragments=None):
        assert input_.ndim == 3
        features = nrag.accumulateEdgeStandartFeatures(self.rag, input_, self.min_value, self.max_value)
        return features[:, self.stat_index]


class FeatureExtractor(object):
    def __init__(self, features_from_filters):
        self.features_from_filters = features_from_filters

    def _boundary_features(self, rag, input_):
        pass

    def _boundary_features_from_filters(self, rag, input_):
        pass

    def boundary_map_features(self, rag, input_):
        return self._boundary_features_from_filters(rag, input_) if self.features_from_filters \
            else self._boundary_features(rag, input_)

    def region_features(self, rag, input_, fragments):
        pass


class RandomForestFeatures(ProblemExtractor):
    def __init__(self, rf_path, features_from_filters=False):
        assert os.path.exists(rf_path), rf_path
        with open(rf_path, 'rb') as f:
            self.rf = pickle.load(f)
        self.feature_extractor = FeatureExtractor(features_from_filters)

    def _compute_edge_probabilities(self, input_, fragments, raw=None):
        features = []
        features.append(self.feature_extractor.boundary_map_features(self.rag, input_))
        if raw is not None:
            features.append(self.feature_extractor.boundary_map_features(self.rag, raw))
            features.append(self.feature_extractor.region_features(self.rag, raw, fragments))
        features = np.concatenate(features, axis=1)
        return self.rf.predict_proba(features)[:, 1]
