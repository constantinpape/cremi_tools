import sys
import numpy as np

sys.path.append('../..')
import cremi_tools.segmentation as cseg
from cremi_tools.viewer.volumina import view


# TODO expose params
def multicut(affs, offsets, solver='kernighan-lin'):
    segmenter = cseg.SegmentationPipeline(cseg.LRAffinityWatershed(0.1, 0.25, 2.),
                                          cseg.MeanAffinityFeatures(offsets),
                                          cseg.Multicut(solver))
    return segmenter(affs)


# TODO
def multicut_From_h5():
    import h5py


def multicut_from_n5(path, raw_key,
                     affinity_key, offsets,
                     bounding_box=(slice(None),),
                     out_key=None):
    import z5py
    affs = z5py.File(path)[affinity_key][(slice(None),) + bounding_box]
    segmentation = multicut(affs, offsets)
    # either save or view the segmentation
    if out_key is None:
        raw = z5py.File(path)[raw_key][bounding_box]
        view([raw, affs.transpose((1, 2, 3, 0)), segmentation])


def default_offsets():
    return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-2, 0, 0], [0, -3, 0], [0, 0, -3],
            [-3, 0, 0], [0, -9, 0], [0, 0, -9],
            [-4, 0, 0], [0, -27, 0], [0, 0, -27]]


if __name__ == '__main__':
    path_to_cremi = ''
    raw_key = ''
    affinity_key = ''
    bounding_box = np.s_[:]
    offsets = default_offsets()
    multicut(path_to_cremi, raw_key, affinity_key, offsets, bounding_box)
