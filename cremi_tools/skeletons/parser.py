import os
import numpy as np


# TODO with bounding box
class SkeletonParser(object):
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


# TODO reactivate if we need these again
# def convert_swc_to_volume(swc_file, shape, resolution,
#                           dtype='uint32',
#                           invert_coordinates=False,
#                           label=1):
#     vol = np.zeros(shape, dtype=dtype)
#     coords = parse_swc_file(swc_file, resolution, invert_coordinates=invert_coordinates, return_where_format=True)
#     vol[coords] = label
#     return vol
#
#
# def paint_in_swc(swc_file, shape, resolution,
#                  invert_coordinates=False,
#                  label=1):
#     from skimage.draw import circle
#     vol = np.zeros(shape, dtype='uint32')
#     coords, radii = parse_swc_file(swc_file, resolution,
#                                    invert_coordinates=invert_coordinates,
#                                    return_where_format=False,
#                                    return_radii=True)
#     for coord, radius in zip(coords, radii):
#         z, y, x = coord
#         rr, cc = circle(y, x, radius, shape=shape[1:])
#         vol[z, rr, cc] = label
#     return vol