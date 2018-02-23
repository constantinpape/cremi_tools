import os
import pickle
import numpy as np
import nifty.graph.rag as nrag

import vigra
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
        min_val, max_val = input_.max(), input_.min()
        return nrag.accumulateEdgeStandartFeatures(rag, input_, min_val, max_val)

    # TODO filters for anisotropic input
    def _boundary_features_from_filters(self, rag, input_):
        filters_ = (filters.gaussianSmoothing,
                    filters.laplacianOfGaussian,
                    filters.hessianOfGaussian)
        sigmas = (1.6, 4.2, 8.4)

        features = []
        for filt in filters_:
            for sigma in sigmas:
                response = filt(input_, sigma)
                if response.ndim == 4:
                    for c in filt.shape[-1]:
                        min_val, max_val = response.min(), response.max()
                        features.append(nrag.accumulateEdgeStandardFeatures(response[..., c],
                                                                            min_val,
                                                                            max_val))
                else:
                    min_val, max_val = response.min(), response.max()
                    features.append(nrag.accumulateEdgeStandardFeatures(response,
                                                                        min_val,
                                                                        max_val))
        return np.concatenate(features, axis=1)

    def boundary_map_features(self, rag, input_):
        return self._boundary_features_from_filters(rag, input_) if self.features_from_filters \
            else self._boundary_features(rag, input_)

    def region_features(self, rag, input_, fragments):
        # list of the region statistics, that we want to extract
        statistics =  [ "Count", "Mean", "Variance",
                        "Skewness", "Kurtosis",
                        "Maximum", "Minimum", "Quantiles",
                        "RegionRadii", "Variance"]

        extractor = vigra.analysis.extractRegionFeatures(input_, fragments,
                                                         features=statistics)
        node_features = np.concatenate([extractor[stat_name][:, None].astype('float32')
                                        if extractor[stat_name].ndim == 1 else extractor[stat_name].astype('float32') for stat_name in statistics],
                                       axis=1)
        uv_ids = rag.uvIds()
        fU = node_features[uv_ids[:,0],:]
        fV = node_features[uv_ids[:,1],:]

        region_features = np.concatenate([np.minimum(fU, fV),
                                          np.maximum(fU, fV),
                                          np.abs(fU - fV),
                                          fU + fV], axis=1)
        return region_features


class RandomForestFeatures(ProblemExtractor):
    def __init__(self, rf_path, features_from_filters=False):
        assert os.path.exists(rf_path), rf_path
        with open(rf_path, 'rb') as f:
            self.rf = pickle.load(f)
        self.feature_extractor = FeatureExtractor(features_from_filters)

    def _compute_edge_probabilities(self, input_, fragments, raw=None, extra_input=None):
        features = []
        features.append(self.feature_extractor.boundary_map_features(self.rag, input_))

        if raw is not None:
            features.append(self.feature_extractor.boundary_map_features(self.rag, raw))
            features.append(self.feature_extractor.region_features(self.rag, raw, fragments))

        if extra_input is not None:
            features.append(self.feature_extractor.boundary_map_features(self.rag, extra_input))

        features = np.concatenate(features, axis=1)
        return self.rf.predict_proba(features)[:, 1]
