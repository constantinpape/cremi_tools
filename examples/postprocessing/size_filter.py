import os
import sys
import numpy as np
import vigra
sys.path.append('../..')
from cremi_tools.postprocessing import merge_small_segments


def size_filter_cremi(sample, size_threshold=100):
    gt = vigra.readHDF5('/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi/sample%s/gt/sample%s_neurongt_automatically_realignedV2.h5'
                        % (sample, sample), 'data')
    hmap = vigra.readHDF5('/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi/sample%s/pmaps/sample%s_affinitiesXY_dunet_atrous_lr_automatically_realigned.h5'
                          % (sample, sample), 'data').astype('float32')

    n_initial = len(np.unique(gt))
    gt_filtered = merge_small_segments(gt, hmap, size_threshold)
    n_filtered = gt_filtered.max() + 1
    print("Reduced number of elements from", n_initial, "to", n_filtered)
    save_path = '/media/papec/data/papec/cremi_gt/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, 'sample%s_neurongt.h5' % sample)
    vigra.writeHDF5(gt_filtered, save_path, 'data', compression='gzip')


if __name__ == '__main__':
    size_filter_cremi('B')
    size_filter_cremi('C')
