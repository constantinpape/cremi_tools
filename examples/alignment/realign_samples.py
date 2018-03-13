import os
import sys
sys.path.append('../..')


def realign_samples(samples):
    from cremi_tools.alignment.realign import realign
    folder = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/'
    for sample in samples:
        raw_path = os.path.join(folder, 'original_data/sample_%s_padded_20160601.hdf' % sample)
        out_path = os.path.join(folder, 'sample%s.h5' % sample)
        realign(raw_path, sample, out_path)


def postprocess(sample):
    from cremi_tools.alignment.realign import postprocess_test_volume
    postprocess_test_volume(sample)


if __name__ == '__main__':
    samples = ('A+', 'B+', 'C+')
    realign_samples(samples)
    # postprocess(samples[0])
