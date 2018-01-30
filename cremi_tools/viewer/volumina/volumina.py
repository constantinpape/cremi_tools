from __future__ import print_function

import sys
import numpy as np
try:
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QColor, QApplication


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
