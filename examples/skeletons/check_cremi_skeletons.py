import os
import sys
import vigra
import h5py
sys.path.append('../..')
from cremi_tools.skeletons import convert_swc_to_volume, paint_in_swc
from cremi_tools.viewer.volumina import view


def check_skeleton(path_to_segmentation, raw_path, swc_folder, obj_id, invert_coordinates=False):
    seg = vigra.readHDF5(path_to_segmentation, 'z/0/data')
    resolution = [35., 4., 4.]
    swc_file = os.path.join(swc_folder, '%04d.swc' % obj_id)
    skel_marked = paint_in_swc(swc_file,
                               seg.shape,
                               resolution,
                               invert_coordinates=invert_coordinates)
    skel_points = convert_swc_to_volume(swc_file,
                                        seg.shape,
                                        resolution,
                                        dtype='uint32',
                                        invert_coordinates=invert_coordinates)
    mask = (seg == obj_id).astype('uint32')
    assert mask.shape == skel_points.shape, "%s, %s" % (str(mask.shape), str(skel_points.shape))
    with h5py.File(raw_path, 'r') as f:
        raw = f['data'][:mask.shape[0]].astype('float32')
        assert raw.shape == mask.shape
    view([raw, mask, skel_marked, skel_points],
         ['raw', 'obj_mask', 'skeleton-marked', 'skeleton-points'])


if __name__ == '__main__':
    seg_path = '/home/papec/mnt/papec/for_dagmar/results_unresolved/splA_z0/result.h5'
    raw_path = '/home/papec/mnt/papec/Work/neurodata_hdd/cremi/sampleA/raw/sampleA_raw_none.h5'
    swc_folder = '/home/papec/mnt/papec/for_dagmar/results_unresolved/splA_z0/skeletons'
    check_skeleton(seg_path, raw_path, swc_folder, 120, invert_coordinates=True)
