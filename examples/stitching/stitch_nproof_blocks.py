import os
import sys
from itertools import combinations

import numpy as np
import vigra

sys.path.append('../..')
from cremi_tools.stitching import stitch_segmentations_by_overlap
try:
    from cremi_tools.viewer.volumina import view
    HAVE_VIEWER = True
except ImportError as e:
    print("Viewer not available")
    HAVE_VIEWER = False


def get_ovlp(id_a, id_b, block_coordinates, blocks, overlaps):
    coords_a, coords_b = block_coordinates[id_a], block_coordinates[id_b]
    shape_a = tuple(ca.stop - ca.start for ca in coords_a)
    shape_b = tuple(cb.stop - cb.start for cb in coords_b)
    # first, we check if the 2 blocks have overlap, by seeing if they have two
    # agreeing coordinates
    equal_axes = tuple(ca.start == cb.start and ca.stop == cb.stop
                       for ca, cb in zip(coords_a, coords_b))
    if sum(equal_axes) != 2:
        return

    # then, we extract the actual overlaps
    ovlp_roi_a = []
    ovlp_roi_b = []
    for ii, is_equal in enumerate(equal_axes):
        if is_equal:
            ovlp_roi_a.append(slice(None))
            ovlp_roi_b.append(slice(None))
        else:
            start_a, start_b = coords_a[ii].start, coords_b[ii].start
            stop_a, stop_b = coords_a[ii].stop, coords_b[ii].stop
            # a is bigger than b
            if stop_a > stop_b:
                assert start_a > start_b
                ovlp_len = stop_b - start_a
                ovlp_roi_a.append(slice(0, ovlp_len))
                ovlp_roi_b.append(slice(shape_b[ii] - ovlp_len, shape_b[ii]))
            else:
                assert start_b > start_a
                ovlp_len = stop_a - start_b
                ovlp_roi_a.append(slice(shape_a[ii] - ovlp_len, shape_a[ii]))
                ovlp_roi_b.append(slice(0, ovlp_len))


    ovlp_a = blocks[id_a][tuple(ovlp_roi_a)]
    ovlp_b = blocks[id_b][tuple(ovlp_roi_b)]
    assert ovlp_a.shape == ovlp_b.shape, "%s, %s" % (str(ovlp_a.shape), str(ovlp_b.shape))

    # view segmentation for debugging
    overlaps[(id_a, id_b)] = (ovlp_a, ovlp_b)


def stitch_fib_blocks(block_folder, out_path, ovlp_threshold=.01):
    # exapmple file path:
    # 'result_x_5000_5520_y_2480_3000_z_3480_4000.h5'

    block_files = os.listdir(block_folder)

    # get the block coordinates
    block_coordinates = []
    for f in block_files:
        split = f.split('_')[1:]
        # get rid of the .h5 ending
        split[-1] = split[-1][:-3]
        split = [int(sp) for sp in split if sp.isdigit()]
        coords = np.s_[split[0]:split[1],
                       split[2]:split[3],
                       split[4]:split[5]]
        block_coordinates.append(coords)

    # load all the blocks
    print("Loading blocks")
    blocks = [vigra.readHDF5(os.path.join(block_folder, f), 'data') for f in block_files]

    # get the overlaps
    print("Extracting overlaps")
    overlaps = {}
    for id_a, id_b in combinations(range(len(blocks)), 2):
        get_ovlp(id_a, id_b, block_coordinates, blocks, overlaps)

    # we expect this to be 12
    print("Number of overlaps:", len(overlaps))

    # run stitching
    print("Running stitcher")
    segmentation = stitch_segmentations_by_overlap(blocks, block_coordinates,
                                                   overlaps,
                                                   ovlp_threshold=ovlp_threshold)

    if HAVE_VIEWER:
        view([segmentation])
    print("Save segmentation")
    # TODO add coordinate offsets to save name
    # out_path += 'x_%i_%i_y_%i_%i_z_%i_%i' % zip()
    vigra.writeHDF5(segmentation, out_path, 'data', compression='gzip')


if __name__ == '__main__':
    block_folder = '/home/papec/Work/neurodata_hdd/fib25/alex_blocks/results_unresolved'
    # block_folder = '/home/papec/Work/neurodata_hdd/fib25/alex_blocks/raw'
    out_path = '/home/papec/Work/neurodata_hdd/fib25/stitched.h5'
    stitch_fib_blocks(block_folder, out_path)
