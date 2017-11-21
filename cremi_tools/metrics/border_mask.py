from concurrent import futures
import numpy as np
from scipy.ndimage import distance_transform_edt


def create_border_mask(segmentation,
                       max_dist,
                       background_label,
                       axis=0,
                       n_threads=8):
    """
    Overlay a border mask with background_label onto input data.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    input_data : np.ndarray - Input data containing neuron ids
    target : np.ndarray - Target which input data overlayed with border mask is written into.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    background_label : int - Border mask will be overlayed using this label.
    axis : int - Axis of iteration (perpendicular to 2d images for which mask will be generated)
    n_threads: number of threads to be used

    Returns
    -------
    segmentation with border mask
    """

    ret = np.zeros_like(segmentation, dtype='uint64')
    sl = [slice(None) for d in range(segmentation.ndim)]

    def border_mask_z(z):
        sl[axis] = z
        border = create_border_mask_2d(segmentation[tuple(sl)], max_dist)
        target_slice = np.copy(segmentation[tuple(sl)])
        target_slice[border] = background_label
        ret[tuple(sl)] = target_slice

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(border_mask_z, z) for z in range(segmentation.shape[axis])]
        [t.result() for t in tasks]
    return ret


def create_border_mask_2d(image, max_dist):
    """
    Create binary border mask for image.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.

    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)

    padded = np.pad(image, 1, mode='edge')

    border_pixels = np.logical_and(np.logical_and(image == padded[:-2, 1:-1],
                                                  image == padded[2:, 1:-1]),
                                   np.logical_and(image == padded[1:-1, :-2],
                                                  image == padded[1:-1, 2:]))

    distances = distance_transform_edt(border_pixels,
                                       return_distances=True,
                                       return_indices=False)
    return distances <= max_dist
