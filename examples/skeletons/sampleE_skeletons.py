import sys
import os
import z5py
import time
import numpy as np

sys.path.append('../..')


def extract_data_central_crop():
    path = '/home/papec/mnt/saalfeldlab/sampleE'
    key = 'segmentations/watershed'

    shape = z5py.File(path)[key].shape
    central = tuple(sh // 2 for sh in shape)
    offset = (100, 1000, 1000)
    bb = tuple(slice(c - off, c + off) for c, off in zip(central, offset))

    ws = z5py.File(path)[key][bb]

    f = z5py.File('./test_ws.n5', use_zarr_format=False)
    ds = f.create_dataset('watershed', shape=ws.shape, dtype='uint64', chunks=(25, 256, 256), compression='gzip')
    ds[:] = ws


def find_skeletons_central_crop(visualize=True, save=False):
    from cremi_tools.skeletons import SkeletonParser
    path = '/home/papec/mnt/nrs/sample_E/sample_E.n5'
    key = 'volumes/raw/s0'

    shape = z5py.File(path)[key].shape

    central = tuple(sh // 2 for sh in shape)

    offset = (100, 1000, 1000)
    bb = tuple(slice(c - off, c + off) for c, off in zip(central, offset))

    # pixel resolution in nanometer
    resolution = (4, 4, 40)
    # sample e offset in nanometer
    offsets = (376000, 80000, 158200)
    parser = SkeletonParser(resolution=resolution,
                            offsets=offsets,
                            invert_coordinates=True)

    t0 = time.time()
    # just an example skeleton for now
    skeleton_folder = '/home/papec/mnt/papec/Work/neurodata_hdd/sampleE/skeletons'
    pn_folder = os.path.join(skeleton_folder, 'v14_PN')
    skeletons = []
    skeleton_ids = []
    skeleton_edges = []
    node_ids = []
    for fpn in os.listdir(pn_folder):
        fpath = os.path.join(pn_folder, fpn)
        skeleton = parser.parse(fpath)
        coords = np.array(skeleton['coordinates'], dtype='uint64')
        nodes = np.array(skeleton['node_ids'], dtype='uint64')
        parents = np.array(skeleton['parents'], dtype='uint64')
        edges = np.concatenate([nodes[:, None], parents[:, None]], axis=1)
        # filter coordinates that are not in sample E
        in_range = np.concatenate([np.logical_and(coords[:, i] >= 0,
                                                  coords[:, i] < shape[i])[:, None] for i in range(3)],
                                  axis=1)
        in_range = np.all(in_range, axis=1)
        skeletons.append(coords[in_range])
        skeleton_ids.append(fpn.split('.')[0])
        # FIXME here, we actually have to check if all the nodes are in range
        skeleton_edges.append(edges[in_range])
        node_ids.append(nodes[in_range])
    t0 = time.time() - t0
    print("Extraction in", t0, "s")

    # intersect the skeletons with our bounding box
    intersecting_skeletons = []
    intersecting_skeleton_ids = []
    intersecting_edges = []
    intersecting_nodes = []
    bb_offset = np.array([bb[i].start for i in range(3)])
    for ii, skel in enumerate(skeletons):
        in_bb = np.concatenate([np.logical_and(skel[:, i] >= bb[i].start,
                                               skel[:, i] < bb[i].stop)[:, None] for i in range(3)],
                               axis=1)
        in_bb = np.all(in_bb, axis=1)
        if in_bb.any():
            intersecting_skeletons.append(skel[in_bb] - bb_offset)
            intersecting_skeleton_ids.append(skeleton_ids[ii])
            # find the intersecting edges by checking which of the potential edges in the bounding
            # box are in range
            nodes = node_ids[ii][in_bb]
            edges = skeleton_edges[ii][in_bb]
            valid_edges = np.in1d(edges, nodes).reshape(edges.shape)
            valid_edges = valid_edges.all(axis=1)
            intersecting_edges.append(edges[valid_edges])
            intersecting_nodes.append(nodes)

    print("Found", len(intersecting_skeletons), "intersecting skeletons")

    # save the skeletons as z5
    if save:
        skel_file = z5py.File('./test_skeletons.n5', use_zarr_format=False)
        for ii, skel_id in enumerate(intersecting_skeleton_ids):
            if skel_id not in skel_file:
                g = skel_file.create_group(skel_id)
            else:
                g = skel_file[skel_id]
            coords = intersecting_skeletons[ii]
            edges = intersecting_edges[ii]
            nodes = intersecting_nodes[ii]

            # we prepend the skeleton node ids to the coordinates
            coords = np.concatenate([nodes[:, None], coords], axis=1)

            coord_ds = g.create_dataset('coordinates', dtype='uint64', shape=coords.shape, chunks=coords.shape)
            coord_ds[:] = coords.astype('uint64')

            # we serialize the edges
            edge_ds = g.create_dataset('edges', dtype='uint64', shape=edges.shape, chunks=edges.shape)
            edge_ds[:] = edges

    if visualize:
        # visualize the skeletons
        print("Make skeleton volume")
        from skimage.draw import circle
        bb_shape = tuple(bb[i].stop - bb[i].start for i in range(3))
        skel_vol = np.zeros(bb_shape, dtype='uint32')
        radius = 10
        for ii, skel in enumerate(intersecting_skeletons):
            for coord in skel:
                z, y, x = coord
                rr, cc = circle(y, x, radius, shape=bb_shape[1:])
                skel_vol[int(z), rr, cc] = ii

        print("Loading raw...")
        raw = z5py.File(path)[key][bb]
        from cremi_tools.viewer.volumina import view
        view([raw, skel_vol])


if __name__ == '__main__':
    find_skeletons_central_crop(True, False)
    # extract_data_central_crop()
