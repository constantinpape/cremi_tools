import os
import numpy as np


# returns coordinates in np.where format
def parse_swc_file(path, resolution,
                   return_radii=False,
                   invert_coordinates=False,
                   return_where_format=False):
    assert os.path.exists(path)
    assert len(resolution) == 3
    coordinates = []
    radii = []
    with open(path, 'r') as f:
        for node in f:
            values = node.split()
            if invert_coordinates:
                coordinates.append([round(float(val) // res) for val, res in zip(values[2:5][::-1], resolution)])
            else:
                coordinates.append([round(float(val) // res) for val, res in zip(values[2:5], resolution)])
            radii.append(float(values[5]) / resolution[1])
    # change to numpy where syntax
    if return_where_format:
        coordinates = tuple(np.array([coords[i] for coords in coordinates], dtype='uint32') for i in range(3))
    else:
        coordinates = [list(map(int, coord)) for coord in coordinates]
    if return_radii:
        return coordinates, radii
    else:
        return coordinates


def convert_swc_to_volume(swc_file, shape, resolution,
                          dtype='uint32',
                          invert_coordinates=False,
                          label=1):
    vol = np.zeros(shape, dtype=dtype)
    coords = parse_swc_file(swc_file, resolution, invert_coordinates=invert_coordinates, return_where_format=True)
    vol[coords] = label
    return vol


def paint_in_swc(swc_file, shape, resolution,
                 invert_coordinates=False,
                 label=1):
    from skimage.draw import circle
    vol = np.zeros(shape, dtype='uint32')
    coords, radii = parse_swc_file(swc_file, resolution,
                                   invert_coordinates=invert_coordinates,
                                   return_where_format=False,
                                   return_radii=True)
    for coord, radius in zip(coords, radii):
        z, y, x = coord
        rr, cc = circle(y, x, radius, shape=shape[1:])
        vol[z, rr, cc] = label
    return vol
