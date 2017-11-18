import sys
sys.path.append('..')
from cremi_tools.alignment.realign import realign_sample, postprocess_test_volume


def realign_samples(samples):
    for sample in samples:
        raw_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sample_%s_padded_20160601.hdf' % sample
        out_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/cremi_warped_sample%s.h5' % sample

        realign_sample(raw_path, sample, out_path)


def postprocess(sample):
        out_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/cremi_warped_sample%s.h5' % sample
        postprocess_test_volume(sample)


if __name__ == '__main__':
    samples = ['C+',]
    realign_samples(samples)
    # postprocess(samples[0])
