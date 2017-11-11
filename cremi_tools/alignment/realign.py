import os
from subprocess import call


def realign_sample(raw_file,
                   sample,
                   out_file,
                   labels_file=None,
                   labels_key=None):
    assert sample in ('A', 'B', 'C', 'A+', 'B+', 'C+')
    assert os.path.exists(raw_file)
    if labels_file is not None:
        assert os.path.exists(raw_file)
        assert labels_key is not None

    # get the directory of this file and get paths to trafos and jar
    this_dir = os.path.dirname(os.path.realpath(__file__))
    transformation = os.path.join(this_dir, 'transformations/sample_%s.transforms.json' % sample)
    assert os.path.exists(transformation)
    jar = os.path.join(this_dir, 'transformations/deform-0.0.1-SNAPSHOT.jar')
    assert os.path.exists(jar)

    cmd = ['java', '-Xmx12g', '-cp', jar,
          'org.janelia.saalfeldlab.deform.DeformToAligned',
          '-i', raw_file,
          '-o', out_file,
          '-t', transformation,
          '-c', '64']
    if labels_file is not None:
        cmd.extend(['-j', labels_file,
                    '-l', labels_key])

    call(cmd)


def rename_key(out_file, old_key='volumes/raw', new_key='data'):
    import h5py
    with h5py.File(out_file) as f:
        f[new_key] = f[old_key]
        del f[old_key]
