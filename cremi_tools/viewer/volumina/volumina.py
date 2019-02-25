#!/usr/bin/python
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
try:
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QColor, QApplication

import h5py
from h5py._hl.dataset import Dataset as H5Dataset

try:
    import z5py
    from z5py.dataset import Dataset as Z5Dataset
except ImportError:
    z5py = None
    Z5Dataset = None


def _name_to_layer(v, d, layer_type, layer_name):
    if layer_type == 'Grayscale':
        v.addGrayscaleLayer(d, name=layer_name)
    elif layer_type == 'RandomColors':
        v.addRandomColorsLayer(d.astype(np.uint32), name=layer_name)
    elif layer_type == 'Red':
        v.addAlphaModulatedLayer(d, name=layer_name, tintColor=QColor(255, 0, 0))
    elif layer_type == 'Green':
        v.addAlphaModulatedLayer(d, name=layer_name, tintColor=QColor(0, 255, 0))
    elif layer_type == 'Blue':
        v.addAlphaModulatedLayer(d, name=layer_name, tintColor=QColor(0, 0, 255))
    else:
        raise KeyError("Invalid Layer Type, %s!" % layer_type)


# get data type of the elements d, to determine
# if we use a grayscale overlay (float32) or a randomcolors overlay (uint) for labels
def _dtype_to_layer(v, d, layer_name):
    data_type = d.dtype
    if data_type == np.float32 or data_type == np.float64 or data_type == np.uint8:
        v.addGrayscaleLayer(d, name=layer_name)
    else:
        v.addRandomColorsLayer(d.astype(np.uint32), name=layer_name)


# plot n data layers
def view(data, labels=None, layer_types=None):
    """
    """

    if labels is not None:
        assert len(labels) == len(data)
    if layer_types is not None:
        assert len(layer_types) == len(data)
        assert all(ltype in ('Grayscale', 'RandomColors', 'Red', 'Green', 'Blue')
                   for ltype in layer_types)

    app = QApplication(sys.argv)
    from volumina.api import Viewer

    v = Viewer()
    v.title = "Volumina Viewer"
    v.showMaximized()

    for i, d in enumerate(data):
        layer_name = layer_name = "layer_" + str(i) if labels is None else labels[i]
        if layer_types is None:
            _dtype_to_layer(v, d, layer_name)
        else:
            _name_to_layer(v, d, layer_types[i], layer_name)

    app.exec_()


def open_file(path):
    ext = os.path.splitext(path)[1]
    if ext.lower in ('.h5', '.hdf5', '.hdf'):
        return h5py.File(path, mode='r')
    elif ext.lower() in ('.n5', '.zr', '.zarr'):
        assert z5py is not None, "z5py not available"
        return z5py.File(path, mode='r')
    else:
        assert False, "Unknown extension: %s" % ext


def append_data(group, data, labels, shape, ndim, prefix):
    axis_reorder = (1, 2, 3, 0) if ndim == 3 else (1, 2, 0)
    for key, obj in group.items():
        name = prefix + '/' + key
        if isinstance(obj, (H5Dataset, Z5Dataset)):
            ds_shape = obj.shape
            # check for compatability of ndim:
            # same number of dimensions -> we can load it
            if len(ds_shape) == ndim:
                if shape is None:
                    shape = ds_shape
                # check that the shapes match, otherwise continue
                if shape != ds_shape:
                    continue
                data.append(obj[:])
                labels.append(name)
            # one dim more -> data with channels, we can load it but need to transpose
            if len(ds_shape) == ndim + 1:
                if shape is None:
                    shape = ds_shape[1:]
                if shape != ds_shape[1:]:
                    continue
                data.append(obj[:].transpose(axis_reorder))
                labels.append(name)
            # non-matching number of dimensions: continue
            else:
                continue
        else:
            append_data(obj, data, labels, shape, ndim, name)


def view_container(path, ndim=3, shape=None):
    """ Display contents of h5 or zarr/n5 container
    """

    assert os.path.exists(path), path
    if shape is not None:
        assert len(shape) == ndim

    data = []
    labels = []

    with open_file(path) as f:
        append_data(f, data, labels, shape, ndim, '')

    view(data, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--ndim', type=int, default=3)
    parser.add_argument('--shape', type=int, nargs='+', default=None)

    args = parser.parse_args()
    view_container(args.path, args.ndim, args.shape)
