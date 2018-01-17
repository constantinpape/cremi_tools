import sys
sys.path.append('../..')
from cremi_tools.viewer.bdv import view


def view_block():
    # path = '/home/papec/mnt/nrs/lauritzen/02/workspace.n5/'
    # keys = ['raw/gray', 'raw/affs_xy_rechunked', 'raw/mask', 'raw/segmentation']
    path = '/home/papec/mnt/saalfeldlab/sampleE/affinity_predictions.n5'
    keys = ['affs_xy']
    inputs = [path] * len(keys)
    view(inputs, keys,
         ranges=[[0, 1]],  # [[0, 255], [0, 1], [0, 1], [0, 100000]],
         resolution=[1, 1, 10])


if __name__ == '__main__':
    view_block()
