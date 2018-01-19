import sys
sys.path.append('../..')
from cremi_tools.viewer.bdv import view


def view_block():
    # path = '/home/papec/mnt/nrs/lauritzen/02/workspace.n5/'
    # keys = ['raw/gray', 'raw/affs_xy_rechunked', 'raw/mask', 'raw/segmentation']
    path = '/media/papec/data/papec/lauritzen_blocks/02_filtered.n5'
    keys = ['gray', 'affs_xy', 'min_filter_mask']
    inputs = [path] * len(keys)
    view(inputs, keys,
         ranges=[[0, 255], [0, 1], [0, 1]],
         resolution=[1, 1, 10])


if __name__ == '__main__':
    view_block()
