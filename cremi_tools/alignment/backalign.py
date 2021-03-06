from __future__ import print_function
import os
from subprocess import call
import numpy as np
import vigra

from ..io import write_custom_key

# value from Stephan's original script:
# which is
#   my orignal crop (23, 1060, 839) for sample A+
#   plus context padding (14, 116, 116)
# which is exactly the gt bounding box crop
offsets = {'A+': (37, 1176, 955),
           'B+': (37, 1076, 1284),
           'C+': (37, 1002, 1165)}


# bounding boxes for the padded inference volumes
# bounding_boxes = {'A+': (slice(23, 176, None), slice(1060, 2821, None), slice(839, 2519, None)),
#                   'B+': (slice(23, 176, None), slice(960, 2893, None), slice(1168, 4194, None)),
#                   'C+': (slice(23, 176, None), slice(886, 2542, None), slice(1049, 2751, None))}

# bounding boxes into fully padded (realigned) volumes necessary before backaligning
bounding_boxes = {'A+': (slice(37, 162, None), slice(1176, 2705, None), slice(955, 2403, None)),
                  'B+': (slice(37, 162, None), slice(1076, 2777, None), slice(1284, 4078, None)),
                  'C+': (slice(37, 162, None), slice(1002, 2426, None), slice(1165, 2635, None))}


def backalign_segmentation(sample, segmentation, out_file,
                           key='volumes/labels/neuron_ids',
                           postprocess=True):

    # if we get a segmentation, we need to write it to a temp file
    cwd = os.getcwd()
    if isinstance(segmentation, np.ndarray):
        seg_path = os.path.join(cwd, 'tmp1.h5')
        write_custom_key(seg_path, segmentation, key)
    elif isinstance(segmentation, str):
        assert os.path.exists(segmentation), segmentation
        seg_path = segmentation
    else:
        raise RuntimeError("Unsupported Input type!")

    exe_file = "transformations/deform-0.0.1-SNAPSHOT.jar"
    trafo_file = "transformations/sample_%s.transforms.json" % sample

    this_dir = os.path.dirname(os.path.realpath(__file__))
    exe_file = os.path.join(this_dir, exe_file)
    trafo_file = os.path.join(this_dir, trafo_file)
    assert os.path.exists(exe_file), exe_file

    source_offset = "%d,%d,%d" % offsets[sample]
    call(["java", "-Xmx6g", "-cp", exe_file, "org.janelia.saalfeldlab.deform.DeformFromAligned",
          "-j", seg_path,  # this is the input file
          "-l", key,
          # offset in input
          # (due to crop of aligned volume by me)
          "--labelssourceoffset", source_offset,
          "-o", out_file,  # this is the output file
          "-t", trafo_file,
          # offset in output
          # (begin of label region in original padded volume, same for all
          # samples)
          "--targetoffset", "37,911,911",
          # interval in which transformations are defined (same for all
          # samples)
          "--transformsize", "200,3072,3072",
          # output size
          "--targetsize", "125,1250,1250",
          # mesh cell size
          "-c", "64"])

    if isinstance(segmentation, np.ndarray):
        os.remove(seg_path)

    if postprocess:
        out = vigra.readHDF5(out_file, key)
        out = vigra.analysis.labelVolumeWithBackground(out)
        os.remove(out_file)
        vigra.writeHDF5(out, out_file, key, compression='gzip')
