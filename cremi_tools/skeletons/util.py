import os
import numpy as np


# TODO support drawing radius from skeletons
def visualize_skeletons(shape, skeletons, radius=5):
    from skimage.draw import circle
    vol = np.zeros(shape, dtype='uint32')
    for skel_id, values in skeletons.items():
        assert 'coordinates' in values
        coords = values['coordinates']
        for coord in coords:
            z, y, x = coord
            rr, cc = circle(y, x, radius, shape=shape[1:])
            vol[z, rr, cc] = skel_id
    return vol


def filter_skeletons_in_rois(skeletons, rois):
    assert isinstance(skeletons, dict)
    assert all(len(roi) == 2 for roi in rois)

    n_skels_filtered = 0
    n_points_filtered = 0
    n_points_total = 0

    filtered_skeletons = {}
    for skeleton_id, skeleton in skeletons.items():
        assert isinstance(skeleton, dict)
        coordinates = skeleton['coordinates']
        edges = skeleton['edges']
        node_ids = skeleton['node_ids']

        initial_len = len(coordinates)
        n_points_total += initial_len

        initial_edges = len(edges)
        assert initial_len == len(node_ids)

        # first we filter out coordinates and corresponding nodes
        # that are in the rois
        valid_mask = np.ones(len(edges), dtype='bool')
        for roi in rois:
            roi_begin, roi_end = roi
            in_roi = [np.logical_and(coordinates[:, ii] >= roi_begin[ii],
                                     coordinates[:, ii] < roi_end[ii])
                      for ii in range(coordinates.shape[1])]
            in_roi = np.logical_and(*in_roi)
            valid_mask[in_roi] = False

        coordinates = coordinates[valid_mask]
        node_ids = node_ids[valid_mask]

        # then we filter edges that connect to nodes that where filtered
        valid_edges = np.in1d(edges, node_ids).reshape(edges.shape).all(axis=1)
        edges = edges[valid_edges]

        if len(coordinates) < initial_len:
            print("Skeleton points in skeleton", skeleton_id, "were filtered")
            print("From", initial_len, "skeleton points to", len(coordinates))
            print("From", initial_edges, "skeleton edges to", len(edges))
            n_skels_filtered += 1
            n_points_filtered += initial_len - len(coordinates)

        # this shouldn't be necessary, because we haven't copied any data
        # and so the values in the skeleton dict should be updated already,
        # but just to be sure ...
        skeleton.update({'coordinates': coordinates,
                         'edges': edges,
                         'node_ids': node_ids})
        filtered_skeletons[skeleton_id] = skeleton

    print("Points were filtered for", n_skels_filtered, "/", len(skeletons), "skeletons")
    print("And the total number of points was filtered by", n_points_filtered, "/", n_points_total)
    return filtered_skeletons


def skeletons_from_csv_to_n5_format(csv_parser, path_to_skeletons):
    """
    Convert skeletons from CSV to n5 format with the given parser.

    Assumes that all skeletons are stored in one csv file.
    """
    assert os.path.exists(path_to_skeletons)
    skeleton_dict = csv_parser.parse(path_to_skeletons)
    with_names = csv_parser.have_name_column
    skel_ids = np.array(skeleton_dict['skeleton_ids'], dtype='uint64')
    node_ids = np.array(skeleton_dict['node_ids'], dtype='uint64')
    parents = np.array(skeleton_dict['parents'], dtype='int64')
    coords = np.array(skeleton_dict['coordinates'], dtype='int64')
    if with_names:
        names = skeleton_dict['names']
    assert (coords >= 0).all()

    # seperate by individual neurons
    extracted_skeletons = {}
    unique_ids = np.unique(skel_ids)
    for skid in unique_ids:
        extracted = {}
        sk_mask = skel_ids == skid

        extracted['coordinates'] = coords[sk_mask]
        if with_names:
            extracted['name'] = names[skid]

        nodes = node_ids[sk_mask]
        parent_nodes = parents[sk_mask]
        edges = np.concatenate([nodes[:, None].astype('int64'),
                                parent_nodes[:, None]], axis=1)
        extracted['node_ids'] = nodes
        extracted['edges'] = edges

        extracted_skeletons[skid] = extracted
    return extracted_skeletons


def skeletons_from_swc_to_n5_format(swc_parser, paths_to_skeletons):
    """
    Convert skeletons from SWC to n5 format with the given parser.

    Assumes that each file contains exactly one skeleton, and that the
    filename has format 'skeleton_id'.swc .
    """
    assert isinstance(paths_to_skeletons, (list, tuple))
    assert all(os.path.exists(path) for path in paths_to_skeletons)

    # separate by individual neurons
    extracted_skeletons = {}
    for path in paths_to_skeletons:
        fname = os.path.split(path)[1].split('.')[0]
        try:
            sk_id = int(fname)
        except ValueError as e:
            raise "Invalid skeleton filename %s" % fname

        values = swc_parser.parse(path)
        nodes = np.array(values['node_ids'], dtype='uint64')
        parents = np.array(values['parents'], dtype='int64')
        edges = np.concatenate([nodes[:, None].astype('int64'),
                                parents[:, None]], axis=1)
        extracted_skeletons[sk_id] = {'coordinates': np.array(values['coordinates'], dtype='int64'),
                                      'node_ids': nodes,
                                      'edges': edges}
    return extracted_skeletons


# serialize the skeletons to n5
def save_skeletons(save_path, save_key, skeletons):
    import z5py

    f = z5py.File(save_path)
    if save_key in f:
        fg = f[save_key]
    else:
        fg = f.create_group(save_key)

    for skel_id, values in skeletons.items():
        g = fg.create_group(str(skel_id))
        # we prepend the nodes to the coordinates
        nodes = values['node_ids']
        coords = values['coordinates']
        coords = np.concatenate([nodes[:, None], coords], axis=1)
        dsc = g.create_dataset('coordinates', shape=coords.shape,
                               chunks=coords.shape, dtype='uint64',
                               compression='raw')
        dsc[:] = coords.astype('uint64')
        # save the edges
        edges = values['edges']
        dse = g.create_dataset('edges', shape=edges.shape,
                               chunks=edges.shape, dtype='int64',
                               compression='raw')
        dse[:] = edges.astype('int64')
        # save the name as attribute if present
        if 'name' in values:
            g.attrs['name'] = values['name']
