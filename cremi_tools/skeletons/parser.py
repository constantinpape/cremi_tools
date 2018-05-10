import os
import csv
import numpy as np


# TODO with bounding box
class SkeletonParserSWC(object):
    """
    Skeleton Parser for .swc exported from CATMAID.
    """
    def __init__(self, resolution, offsets, invert_coordinates=False):
        assert len(resolution) == len(offsets) == 3
        self.resolution = resolution
        self.offsets = offsets
        self.invert_coordinates = invert_coordinates

    def parse(self, skeleton_file, return_where_format=False):
        assert os.path.exists(skeleton_file), skeleton_file
        node_ids = []
        coordinates = []
        parents = []
        radii = []
        with open(skeleton_file, 'r') as f:
            for node in f:
                values = node.split()
                # extract coordinate
                coord = [round((float(val) - off) // res)
                         for val, off, res in zip(values[2:5], self.offsets, self.resolution)]
                if self.invert_coordinates:
                    coord = coord[::-1]
                coordinates.append(coord)
                # extract radius
                radii.append(float(values[5]) / self.resolution[1])
                # extract node id and parent
                node_ids.append(int(values[0]))
                parents.append(int(values[-1]))
        if return_where_format:
            coordinates = tuple(np.array([coords[i] for coords in coordinates], dtype='uint32') for i in range(3))
        else:
            coordinates = [list(map(int, coord)) for coord in coordinates]
        return {'node_ids': node_ids, 'coordinates': coordinates, 'parents': parents, 'radii': radii}


# TODO with bounding box
class SkeletonParserCSV(object):
    """
    Skeleton Parser for .csv exported from CATMAID.
    """
    def __init__(self, resolution, offsets, invert_coordinates=False, have_name_column=True):
        assert len(resolution) == len(offsets) == 3
        self.resolution = resolution
        self.offsets = offsets
        self.invert_coordinates = invert_coordinates
        self.have_name_column = have_name_column

    def _parse_with_name(self, skeleton_file, return_where_format):
        skeleton_names = {}
        skeleton_ids = []
        node_ids = []
        coordinates = []
        parents = []
        radii = []

        last_id = -1

        with open(skeleton_file, 'r', newline='') as f:
            skelreader = csv.reader(f, delimiter=',')

            # skip the header
            next(skelreader, None)

            for values in skelreader:

                # find skeleton ids and names (we only extract a dict from id to name)
                skel_id = int(values[1])
                skeleton_ids.append(skel_id)
                if skel_id != last_id:
                    skeleton_names[skel_id] = values[0]
                last_id = skel_id

                # extract node id and parent
                node_ids.append(int(values[2]))
                if values[3] == '':
                    parents.append(-1)
                else:
                    parents.append(int(values[3]))

                # extract coordinate
                coord = [round((float(val) - off) // res)
                         for val, off, res in zip(values[4:7], self.offsets, self.resolution)]
                if self.invert_coordinates:
                    coord = coord[::-1]
                coordinates.append(coord)
                # extract radius
                radii.append(float(values[7]) / self.resolution[1])

        if return_where_format:
            coordinates = tuple(np.array([coords[i] for coords in coordinates], dtype='uint32')
                                for i in range(3))
        else:
            coordinates = [list(map(int, coord)) for coord in coordinates]

        return {'names': skeleton_names,
                'skeleton_ids': skeleton_ids,
                'node_ids': node_ids,
                'coordinates': coordinates,
                'parents': parents,
                'radii': radii}

    def _parse_without_name(self, skeleton_file, return_where_format):
        skeleton_ids = []
        node_ids = []
        coordinates = []
        parents = []
        radii = []

        with open(skeleton_file, 'r', newline='') as f:
            skelreader = csv.reader(f, delimiter=',')

            # skip the header
            next(skelreader, None)

            for values in skelreader:

                # find skeleton ids and names (we only extract a dict from id to name)
                skel_id = int(values[0])
                skeleton_ids.append(skel_id)

                # extract node id and parent
                node_ids.append(int(values[1]))
                if values[2] == '' or values[2] == 'undefined':
                    parents.append(-1)
                else:
                    # print(values[2])
                    parents.append(int(values[2]))

                # extract coordinate
                coord = [round((float(val) - off) // res)
                         for val, off, res in zip(values[3:6], self.offsets, self.resolution)]
                if self.invert_coordinates:
                    coord = coord[::-1]
                coordinates.append(coord)
                # extract radius
                radii.append(float(values[6]) / self.resolution[1])

        if return_where_format:
            coordinates = tuple(np.array([coords[i] for coords in coordinates], dtype='uint32')
                                for i in range(3))
        else:
            coordinates = [list(map(int, coord)) for coord in coordinates]

        return {'skeleton_ids': skeleton_ids,
                'node_ids': node_ids,
                'coordinates': coordinates,
                'parents': parents,
                'radii': radii}

    def parse(self, skeleton_file, return_where_format=False):
        assert os.path.exists(skeleton_file), skeleton_file
        return self._parse_with_name(skeleton_file, return_where_format) if self.have_name_column else \
            self._parse_without_name(skeleton_file, return_where_format)


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
