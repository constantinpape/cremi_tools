import sys
sys.path.append('..')
from cremi_tools.alignment.realign import realign_sample, rename_key


def realign_samples(samples):
    for sample in samples:
        raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample_%s_padded_20160601.hdf' % sample
        out_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/cremi_warped_sample%s.h5' % sample

        realign_sample(raw_path, sample, out_path)
        rename_key(out_path)


if __name__ == '__main__':
    samples = ['A+',]
    realign_samples(samples)
