import h5py
import z5py

from cremi_tools.alignment import backalign_segmentation, bounding_boxes
from cremi_tools.viewer.volumina import view


def check_backalign():
    path1 = '/home/pape/Work/data/cremi/sampleA+.h5'
    bb = bounding_boxes['A+']

    # print("Loading segmentation ...")
    # with h5py.File(path1) as f:
    #     ds = f['segmentation/multicut']
    #     seg = ds[bb]

    tmp_path = './A+_realigned_tmp.h5'
    # print("Writing segmentation ...")
    # with h5py.File(tmp_path) as f:
    #     f.create_dataset('volumes/labels/neuron_ids', data=seg, compression='gzip')

    backalign_segmentation('A+', tmp_path, './A+_tmp.h5',
                           postprocess=False)


if __name__ == '__main__':
    check_backalign()
