import multiprocessing
# from concurrent import futures
import numpy as np
import numpy.masked_array as ma
import vigra
from scipy.ndimage.morphology import distance_transform_edt

try:
    import constrained_mst as cmst
    WITH_CMST = True
except ImportError as e:
    WITH_CMST = False

from .base import Oversegmenter


def seeds_from_connected_components(input_, threshold, exclusion_mask=None):
    # generate seeds from thresholded connected components
    thresholded = input_ <= threshold
    if exclusion_mask is not None:
        thresholded[exclusion_mask] = 0
    seeds = vigra.analysis.labelVolumeWithBackground(thresholded.view('uint8'))
    max_label = int(seeds.max())
    return seeds, max_label + 1


def seeds_from_distance_transform_2d(input_, threshold, sigma):
    thresholded = input_ < threshold
    seeds = np.zeros_like(thresholded, dtype='uint32')
    offset_z = 0
    for z in range(seeds.shape[0]):
        dt = distance_transform_edt(thresholded[z]).astype('float32')
        if sigma > 0.:
            dt = vigra.filters.gaussianSmoothing(dt, sigma)
        seeds_z = vigra.analysis.localMaxima(dt, allowPlateaus=True, allowAtBorder=True, marker=np.nan)
        seeds_z = vigra.analysis.labelImageWithBackground(np.isnan(seeds_z).view('uint8'))
        seeds_z[seeds_z != 0] += offset_z
        offset_z = seeds_z.max() + 1
        seeds[z] = seeds_z
    return seeds, offset_z


def seeds_from_distance_transform(input_, threshold, sigma):
    thresholded = input_ < threshold
    seeds = np.zeros_like(thresholded, dtype='uint32')
    dt = distance_transform_edt(thresholded).astype('float32')
    if sigma > 0.:
        dt = vigra.filters.gaussianSmoothing(dt, sigma)
    seeds = vigra.analysis.localMaxima3D(dt, allowPlateaus=True, allowAtBorder=True, marker=np.nan)
    seeds = vigra.analysis.labelImageWithBackground(np.isnan(seeds).view('uint8'))
    return seeds, seeds.max() + 1


def size_filter(hmap, ws, size_filter=25):
    ids, sizes = np.unique(ws, return_counts=True)
    mask = np.ma.masked_array(ws, np.in1d(ws, ids[sizes < size_filter])).mask
    ws[mask] = 0
    ws, max_id = vigra.analysis.watershedsNew(hmap, seeds=ws)
    return ws, max_id


def run_watershed_2d(hmap, seeds):
    # run watersheds in 2d
    ws = np.zeros_like(seeds, dtype='uint32')
    for z in range(ws.shape[0]):
        ws[z] = vigra.analysis.watershedsNew(hmap[z], seeds=seeds[z])[0]
    return ws, int(ws.max())


def run_watershed(hmap, seeds):
    return vigra.analysis.watershedsNew(hmap, seeds=seeds)


# TODO support different isotropy settings for distance transform
def filter_with_dt(input_, seeds, seed_offset,
                   threshold_dt, threshold_filter,
                   is_anisotropic):
    thresholded = input_ < threshold_dt

    if is_anisotropic:
        dt = distance_transform_edt(thresholded)
    else:
        dt = np.zeros_like(thresholded, dtype='float32')
        for z in range(dt.shape[0]):
            dt[z] = distance_transform_edt(thresholded[z])

    filter_map = dt > threshold_filter
    filter_map = vigra.analysis.labelVolumeWithBackground(filter_map.view('uint8'))

    # TODO this is pretty inefficient, if this idea works out, we need to move it to cpp ...
    seed_ids = np.unique(seeds)[1:]
    for seed_id in seed_ids:
        where_seed = seeds == seed_id
        where_filtered = filter_map[where_seed]
        filtered_ids_in_seed = np.unique(where_filtered)
        if filtered_ids_in_seed[0] == 0:
            filtered_ids_in_seed = filtered_ids_in_seed[1:]

        # if we have more than one distance transform component,
        # we reject this seed and replace it with the components
        if len(filtered_ids_in_seed) > 1:
            where_filtered[where_filtered != 0] += seed_offset
            seeds[where_seed] = where_filtered

    return seeds, seeds.max() + 1


class LRAffinityWatershed(Oversegmenter):
    def __init__(self, threshold_cc, threshold_dt, sigma_seeds, size_filter=25,
                 threshold_filter=None, is_anisotropic=True,
                 seed_channel=None, **super_kwargs):
        super(LRAffinityWatershed, self).__init__(**super_kwargs)
        self.threshold_cc = threshold_cc
        self.threshold_dt = threshold_dt
        self.sigma_seeds = sigma_seeds
        self.size_filter = size_filter
        self.threshold_filter = threshold_filter
        self.is_anisotropic = is_anisotropic
        if seed_channel is not None:
            assert isinstance(seed_channel, list)
            self.seed_channel = seed_channel
        else:
            self.seed_channel = None

    def _oversegmentation_impl(self, input_):
        assert input_.ndim == 4
        full = np.mean(input_, axis=0) if self.seed_channel is None else np.mean(input_[self.seed_channel], axis=0)
        nn_slice = slice(1, 3) if self.is_anisotropic else slice(0, 3)
        nearest = np.mean(input_[nn_slice], axis=0)

        seeds, seed_offset = seeds_from_connected_components(full, self.threshold_cc)

        # if we have a filter threshold, we additionally calculate connected
        # components on the distance transofrm of the full map
        if self.threshold_filter is not None:

            if self.return_seeds is not None:
                unfiltered = seeds.copy()

            seeds, seed_offset = filter_with_dt(full, seeds, seed_offset,
                                                self.threshold_dt, self.threshold_filter)

        if self.is_anisotropic:
            seeds_dt, _ = seeds_from_distance_transform_2d(nearest,
                                                           self.threshold_dt,
                                                           self.sigma_seeds)
        else:
            seeds_dt, _ = seeds_from_distance_transform(nearest,
                                                        self.threshold_dt,
                                                        self.sigma_seeds)

        # merge seeds
        seeds_dt[seeds_dt != 0] += seed_offset
        no_seed_mask = seeds == 0
        seeds[no_seed_mask] = seeds_dt[no_seed_mask]

        # run watershed
        if self.is_anisotropic:
            ws, max_id = run_watershed_2d(nearest, seeds)
        else:
            ws, max_id = run_watershed(nearest, seeds)
        if size_filter:
            ws, max_id = size_filter(nearest, ws, self.size_filter)

        if self.return_seeds and self.threshold_filter is not None:
            return ws, max_id, seeds, unfiltered
        elif self.return_seeds:
            return ws, max_id, seeds
        else:
            return ws, max_id

    def _oversegmentation_impl_masked(self, input_, mask):
        assert input_.ndim == 4
        full = np.mean(input_, axis=0) if self.seed_channel is None else np.mean(input_[self.seed_channel], axis=0)
        nn_slice = slice(1, 3) if self.is_anisotropic else slice(0, 3)
        nearest = np.mean(input_[nn_slice], axis=0)
        # get the excluded area (= inverted mask)
        exclusion_mask = np.logical_not(mask)

        seeds, seed_offset = seeds_from_connected_components(full,
                                                             self.threshold_cc,
                                                             exclusion_mask)

        # if we have a filter threshold, we additionally calculate connected
        # components on the distance transofrm of the full map
        if self.threshold_filter is not None:

            if self.return_seeds is not None:
                unfiltered = seeds.copy()

            full[exclusion_mask] = 1
            seeds, seed_offset = filter_with_dt(full, seeds, seed_offset,
                                                self.threshold_dt, self.threshold_filter,
                                                self.is_anisotropic)

        # mask excluded area in the grow map
        nearest[exclusion_mask] = 1
        if self.is_anisotropic:
            seeds_dt, _ = seeds_from_distance_transform_2d(nearest,
                                                           self.threshold_dt,
                                                           self.sigma_seeds)
        else:
            seeds_dt, _ = seeds_from_distance_transform(nearest,
                                                        self.threshold_dt,
                                                        self.sigma_seeds)

        # merge seeds
        seeds_dt[seeds_dt != 0] += seed_offset
        no_seed_mask = seeds == 0
        seeds[no_seed_mask] = seeds_dt[no_seed_mask]

        # run watershed
        if self.is_anisotropic:
            ws, max_id = run_watershed_2d(nearest, seeds)
        else:
            ws, max_id = run_watershed(nearest, seeds)

        if size_filter:
            ws, max_id = size_filter(nearest, ws, self.size_filter)

        ws[exclusion_mask] = 0
        ws, max_id, _ = vigra.analysis.relabelConsecutive(ws, keep_zeros=True)

        if self.return_seeds and self.threshold_filter is not None:
            return ws, max_id, seeds, unfiltered
        elif self.return_seeds:
            return ws, max_id, seeds
        else:
            return ws, max_id


class DTWatershed(Oversegmenter):
    def __init__(self, threshold_dt, sigma_seeds, size_filter=25,
                 is_anisotropic=True, n_threads=-1, **super_kwargs):
        super(LRAffinityWatershed, self).__init__(**super_kwargs)
        self.threshold_dt = threshold_dt
        self.sigma_seeds = sigma_seeds
        self.size_filter = size_filter
        self.is_anisotropic = is_anisotropic
        self.n_threads = multiprocessing.cpu_count() if n_threads == -1 else n_threads

    # TODO
    def _oversegmentation_impl(self, input_):
        assert input_.ndim == 3

    # TODO
    def _oversegmentation_impl_masked(self, input_, mask):
        assert input_.ndim == 3


class MutexWatershed(Oversegmenter):
    def __init__(self, offsets, strides,
                 seperating_channel=3,
                 invert_repulsive_channels=True,
                 randomize_bounds=True):
        assert WITH_CMST
        assert isinstance(offsets, list)
        assert all(isinstance(off, list) for off in offsets)
        self.offsets = offsets
        assert isinstance(strides, list)
        assert all(isinstance(stride, list) for stride in strides)
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
        sorted_edges = self._sorted_edges(input_)
        # sorted_edges = np.argsort(input_.ravel())

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
        sorted_edges = self._sorted_edges(input_)
        # sorted_edges = np.argsort(input_.ravel())

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
        return segmentation
