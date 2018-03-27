import sys
sys.path.append('../..')


def make_label_multiset(block_id, in_key, out_key):
    from cremi_tools.sampling import create_label_multiset
    chunks = [256, 256, 26]
    input_path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5' % block_id
    key1 = '/'.join(('filtered', 'segmentations', in_key))
    key2 = '/'.join(('filtered', 'segmentations', out_key))
    create_label_multiset(input_path, key1, key2, chunks)


def downsample(block_id, in_key):
    from cremi_tools.sampling import downsample_labels
    input_path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5' % block_id
    key = '/'.join(('filtered', 'segmentations', in_key))

    sampling_factors = [[2, 2, 1],
                        [2, 2, 1],
                        [2, 2, 1],
                        [2, 2, 2]]
    block_sizes = [[256, 256, 26],
                   [128, 128, 26],
                   [128, 128, 26],
                   [128, 128, 13]]
    m_best = [5]
    downsample_labels(input_path, key,
                      sampling_factors,
                      block_sizes,
                      m_best)


if __name__ == '__main__':
    block_id = 2
    in_key = 'mc_more_features_merged'
    out_key = in_key + '_multiscale'
    # make_label_multiset(block_id, in_key, out_key)
    downsample(block_id, out_key)
