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
    def __init__(self, resolution, offsets, invert_coordinates=False):
        assert len(resolution) == len(offsets) == 3
        self.resolution = resolution
        self.offsets = offsets
        self.invert_coordinates = invert_coordinates

    def parse(self, skeleton_file, return_where_format=False):
        assert os.path.exists(skeleton_file), skeleton_file
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
