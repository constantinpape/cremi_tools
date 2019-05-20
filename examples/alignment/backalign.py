import h5py
import z5py

from cremi_tools.alignment import backalign_segmentation, bounding_boxes
from cremi_tools.viewer.volumina import view


def check_backalign(sample):
    path1 = '/g/kreshuk/data/cremi/padded_realigned/sample%s.n5' % sample
    bb = bounding_boxes[sample]
    halo = [14, 116, 116]
    bb = tuple(slice(b.start + ha, b.stop - ha) for b, ha in zip(bb, halo))
    print(bb)

    print("Loading segmentation ...")
    with z5py.File(path1) as f:
        ds = f['segmentation/multicut']
        seg = ds[bb]

    tmp_path = './%s_realigned_tmp.h5' % sample
    print("Writing segmentation ...")
    with h5py.File(tmp_path) as f:
        f.create_dataset('volumes/labels/neuron_ids', data=seg, compression='gzip')

    backalign_segmentation(sample, tmp_path, './%s_tmp.h5' % sample,
                           postprocess=False)


def view_backaligned(sample):
    path = '/g/kreshuk/data/cremi/original/sample_%s_20160601.hdf' % sample
    with h5py.File(path) as f:
        raw = f['volumes/raw'][:]

    with h5py.File('./%s_tmp.h5' % sample) as f:
        seg = f['volumes/labels/neuron_ids'][:]
    assert raw.shape == seg.shape

    view([raw, seg])


if __name__ == '__main__':
    sample = 'C+'
    check_backalign(sample)
    view_backaligned(sample)
