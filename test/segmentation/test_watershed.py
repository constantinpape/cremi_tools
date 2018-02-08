import sys
import unittest
import z5py
import numpy as np

sys.path.append('../..')
import cremi_tools.segmentation as cseg
from cremi_tools.viewer.volumina import view


class TestWatershed(unittest.TestCase):
    # TODO data without mask
    path = '/home/papec/Work/neurodata_hdd/testdata.n5'
    bb = np.s_[:, :]
    # bb = np.s_[:20, :756, :756]

    def _test_wslr(self):
        bb_aff = (slice(None),) + self.bb
        affs = z5py.File(self.path)['full_affs'][bb_aff]
        segmenter = cseg.LRAffinityWatershed(0.1, 0.25, 2.,
                                             return_seeds=True,
                                             invert_input=True)
        print("Start watershed...")
        ws, seeds, _ = segmenter(affs)
        print("... done")
        raw = z5py.File(self.path)['raw'][self.bb]
        view([raw, 1. - affs.transpose((1, 2, 3, 0)), seeds, ws],
             ['raw', 'affs', 'seeds', 'ws'])

    def test_wslr_masked(self):
        bb_aff = (slice(None),) + self.bb
        affs = z5py.File(self.path)['full_affs'][bb_aff]
        mask = z5py.File(self.path)['mask'][self.bb]
        segmenter = cseg.LRAffinityWatershed(0.1, 0.25, 2.,
                                             return_seeds=True,
                                             invert_input=True)
        print("Start watershed...")
        ws, seeds, _ = segmenter(affs, mask)
        print("... done")

        if True:
            f = z5py.File(self.path)
            ds = f.create_dataset('watershed', shape=ws.shape, chunks=(25, 256, 256),
                                  dtype='uint64', compression='gzip')
            ds[:] = ws.astype('uint64')

        raw = z5py.File(self.path)['raw'][self.bb]
        view([raw, mask, 1. - affs.transpose((1, 2, 3, 0)), seeds, ws],
             ['raw', 'mask', 'affs', 'seeds', 'ws'])


if __name__ == '__main__':
    unittest.main()
