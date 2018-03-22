import multiprocessing
# from concurrent import futures
import numpy as np
import vigra
from scipy.ndimage.morphology import distance_transform_edt

from .base import Oversegmenter


def seeds_from_connected_components(input_, threshold, exclusion_mask=None):
    # generate seeds from thresholded connected components
    thresholded = input_ <= threshold
    if exclusion_mask is not None:
        thresholded[exclusion_mask] = 0
    seeds = vigra.analysis.labelVolumeWithBackground(thresholded.view('uint8'))
    max_label = int(seeds.max())
    return seeds, max_label + 1


def seeds_from_zero_components(input_, sigma, exclusion_mask=None):
    # generate seeds from the zero component, after smoothing
    if sigma > 0:
        seed_map = vigra.filters.gaussianSmoothing(input_, sigma)
    else:
        seed_map = input_.copy()
    seed_map -= seed_map.min()
    seed_map = seed_map == 0

    if exclusion_mask is not None:
        [exclusion_mask] = 0

    seeds = vigra.analysis.labelVolumeWithBackground(seed_map.view('uint8'))
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


def filter_by_size(hmap, ws, size_filter=25):
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


class LRAffinityWatershed(Oversegmenter):
    def __init__(self, threshold_cc, threshold_dt, sigma_seeds, size_filter=25,
                 is_anisotropic=True, seed_channel=None, **super_kwargs):
        super(LRAffinityWatershed, self).__init__(**super_kwargs)
        self.threshold_cc = threshold_cc
        self.threshold_dt = threshold_dt
        self.sigma_seeds = sigma_seeds
        self.size_filter = size_filter
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

        if self.threshold_cc > 0:
            seeds, seed_offset = seeds_from_connected_components(full, self.threshold_cc)
        else:
            seeds, seed_offset = seeds_from_zero_components(full, self.sigma_seeds)

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
        if self.size_filter:
            ws, max_id = filter_by_size(nearest, ws, self.size_filter)

        if self.return_seeds:
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

        if self.threshold_cc > 0:
            seeds, seed_offset = seeds_from_connected_components(full,
                                                                 self.threshold_cc,
                                                                 exclusion_mask)
        else:
            seeds, seed_offset = seeds_from_zero_components(full,
                                                            self.sigma_seeds,
                                                            exclusion_mask)

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

        if self.size_filter:
            ws, max_id = filter_by_size(nearest, ws, self.size_filter)

        ws[exclusion_mask] = 0
        ws, max_id, _ = vigra.analysis.relabelConsecutive(ws, keep_zeros=True)

        if self.return_seeds:
            return ws, max_id, seeds
        else:
            return ws, max_id


class DTWatershed(Oversegmenter):
    def __init__(self, threshold_dt, sigma_seeds, size_filter=25,
                 is_anisotropic=True, n_threads=-1, **super_kwargs):
        super(DTWatershed, self).__init__(**super_kwargs)
        self.threshold_dt = threshold_dt
        self.sigma_seeds = sigma_seeds
        self.size_filter = size_filter
        self.is_anisotropic = is_anisotropic
        self.n_threads = multiprocessing.cpu_count() if n_threads == -1 else n_threads

    def _oversegmentation_impl(self, input_):
        assert input_.ndim == 3

        if self.is_anisotropic:
            seeds, _ = seeds_from_distance_transform_2d(input_,
                                                        self.threshold_dt,
                                                        self.sigma_seeds)
        else:
            seeds, _ = seeds_from_distance_transform(input_,
                                                     self.threshold_dt,
                                                     self.sigma_seeds)

        if self.is_anisotropic:
            ws, max_id = run_watershed_2d(input_, seeds)
        else:
            ws, max_id = run_watershed(input_, seeds)
        if self.size_filter:
            ws, max_id = filter_by_size(input_, ws, self.size_filter)

        return ws, max_id

    def _oversegmentation_impl_masked(self, input_, mask):
        assert input_.ndim == 3

        # get the excluded area (= inverted mask)
        exclusion_mask = np.logical_not(mask)
        # mask excluded area in the grow map
        input_[exclusion_mask] = 1

        if self.is_anisotropic:
            seeds, _ = seeds_from_distance_transform_2d(input_,
                                                        self.threshold_dt,
                                                        self.sigma_seeds)
        else:
            seeds, _ = seeds_from_distance_transform(input_,
                                                     self.threshold_dt,
                                                     self.sigma_seeds)

        if self.is_anisotropic:
            ws, max_id = run_watershed_2d(input_, seeds)
        else:
            ws, max_id = run_watershed(input_, seeds)

        if self.size_filter:
            ws, max_id = filter_by_size(input_, ws, self.size_filter)

        ws[exclusion_mask] = 0
        ws, max_id, _ = vigra.analysis.relabelConsecutive(ws, keep_zeros=True)

        return ws, max_id
