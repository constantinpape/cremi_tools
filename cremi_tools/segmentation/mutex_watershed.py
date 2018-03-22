import numpy as np
import numpy.ma as ma
from .base import Oversegmenter

try:
    import constrained_mst as cmst
    WITH_CMST = True
except ImportError as e:
    WITH_CMST = False


# TODO nifty mutex watershed
class MutexWatershed(Oversegmenter):
    def __init__(self, offsets, strides,
                 seperating_channel=3,
                 invert_repulsive_channels=True,
                 randomize_bounds=True,
                 **super_kwargs):
        super(MutexWatershed, self).__init__(**super_kwargs)
        assert WITH_CMST
        assert isinstance(offsets, list)
        assert all(isinstance(off, list) for off in offsets)
        self.offsets = offsets

        assert isinstance(strides, list)
        self.strides = strides
        self.seperating_channel = seperating_channel
        self.invert_repulsive_channels = invert_repulsive_channels
        self.randomize_bounds = randomize_bounds

    # TODO randomize bounds ?!
    def _sort_edges(self, input_):
        mask = np.ones(input_.shape)
        # use all attractive edges
        mask[:self.seperating_channel] = 0
        # subsample repulsive edges
        if len(self.strides) == 3:
            mask[self.seperating_channel:,
                 ::self.strides[0],
                 ::self.strides[1],
                 ::self.strides[2]] = 1
        else:
            raise NotImplementedError()
        masked_edges = ma.array(input_, mask=mask, fill_value=999)
        # TODO: compute n_used_edges analytically from strides
        n_used_edges = mask.size - mask.sum()
        # axis=None sorts the flattend array
        return masked_edges.argsort(axis=None)[:n_used_edges]

    # TODO subsample before sorting ?!
    def _oversegmentation_impl(self, input_):
        assert input_.ndim == 4
        assert len(input_) == len(self.offsets), "%s, %i" % (str(input_.shape), len(self.offsets))
        if self.invert_repulsive_channels:
            input_[self.seperating_channel:] *= -1
            input_[self.seperating_channel:] += 1

        # TODO make sure the pre-sorting works
        # sorted_edges = self._sorted_edges(input_)
        sorted_edges = np.argsort(input_.ravel())

        # run the mst watershed
        mst = cmst.ConstrainedWatershed(np.array(input_.shape[1:]),
                                        self.offsets,
                                        self.seperating_channel,
                                        np.array(self.strides))
                                        # np.array([1, 1, 1]))
        # self.strides)  # need [1, 1, 1] strides if we subsample before sorting

        # don't need this if we subsample before sorting
        # if self.randomize_bounds:
        #     mst.compute_randomized_bounds()
        mst.repulsive_ucc_mst_cut(sorted_edges, 0)
        segmentation = mst.get_flat_label_image().reshape(input_.shape[1:])
        return segmentation

    # TODO how do we mask ?!
    def _oversegmentation_impl_masked(self, input_, mask):
        assert input_.ndim == 4
        assert input_.ndim == 4
        assert len(input_) == len(self.offsets), "%s, %i" % (str(input_.shape), len(self.offsets))
        if self.invert_repulsive_channels:
            input_[self.seperating_channel:] *= -1
            input_[self.seperating_channel:] += 1

        # we mask by setting the local / attractive affinity channels to 1, so they will never be drawn
        # TODO is this correct ?
        input_[:3, mask] = 1

        # TODO make sure the pre-sorting works
        # sorted_edges = self._sorted_edges(input_)
        sorted_edges = np.argsort(input_.ravel())

        # run the mst watershed
        mst = cmst.ConstrainedWatershed(np.array(input_.shape[1:]),
                                        self.offsets,
                                        self.seperating_channel,
                                        np.array([1, 1, 1]))
        # self.strides)  # need [1, 1, 1] strides if we subsample before sorting

        # don't need this if we subsample before sorting
        # if self.randomize_bounds:
        #     mst.compute_randomized_bounds()
        mst.repulsive_ucc_mst_cut(sorted_edges, 0)
        segmentation = mst.get_flat_label_image().reshape(input_.shape[1:])
        return segmentation, int(segmentation.max())
