import os
import sys
from itertools import combinations

import numpy as np
import vigra

sys.path.append('../..')
from cremi_tools.stitching import stitch_segmentations_by_overlap


def get_ovlp(id_a, id_b, block_coordinates, blocks, overlaps):
    coords_a, coords_b = block_coordinates[id_a], block_coordinates[id_b]
    # first, we check if the 2 blocks have overlap, by seeing if they have a
    # single coordinate with
    center_a = tuple((c.start + c.stop) // 2 for c in coords_a)
    center_b = tuple((c.start + c.stop) // 2 for c in coords_b)
    diff = tuple(abs(ca - cb) for ca, cb in zip(center_a, center_b))
    ovlp_dim = tuple(d for d in diff if d == 0)
    if not len(ovlp_dim) == 1:
        return

    # then, we extract the actual overlaps


def stitch_fib_blocks(block_folder, out_path, ovlp_threshold=.9):
    # exapmple file path:
    # 'result_x_5000_5520_y_2480_3000_z_3480_4000.h5'

    block_files = os.listdir(block_folder)

    # get the block coordinates
    block_coordinates = []
    for f in block_files:
        split = f.split('_')[1:]
        split = [int(sp) for sp in split if sp.isdigit()]
        coords = np.s_[split[0]:split[1],
                       split[2]:split[3],
                       split[4]:split[5]]
        block_coordinates.append(coords)

    # load all the blocks
    blocks = [vigra.readHDF5(os.path.join(block_folder, f), 'data') for f in block_files]

    # get the overlaps
    overlaps = {}
    for id_a, id_b in combinations(range(len(blocks)), 2):
        get_ovlp(id_a, id_b, block_coordinates, blocks, overlaps)

    # TODO how much do we expect for 8 blocks ?
    print("NUmber of overlaping blocks:", len(overlaps))

    # run stitching
    stitch_segmentations_by_overlap(blocks, block_coordinates,
                                    overlaps,
                                    out_path, 'data',
                                    ovlp_threshold=ovlp_threshold)


if __name__ == '__main__':
    block_folder = '/home/constantin/Work/neurodata_hdd/FIB25/alex_results'
    out_path = ''
    stitch_fib_blocks(block_folder, out_path)
