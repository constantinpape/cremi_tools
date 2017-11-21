from __future__ import print_function
import h5py


# TODO check if this is really the default resolution
def add_resolution_tag(ds):
    ds.attrs['resolution'] = (40., 4., 4.)


def write_neuron_ids(save_file, data):
    key = 'volumes/labels/neuron_ids'
    with h5py.File(save_file) as f:
        if key in f:
            print("Overwriting existing NeuronIds")
            f[key] = data
        else:
            f.create_dataset(key, data=data, compression='gzip', dtype='uint64')
        # add resolution tag
        add_resolution_tag(f[key])


def write_clefts(save_file, data):
    key = 'volumes/labels/clefts'
    with h5py.File(save_file) as f:
        if key in f:
            print("Overwriting existing Clefts")
            f[key] = data
        else:
            f.create_dataset(key, data=data, compression='gzip', dtype='uint64')
        # add resolution tag
        add_resolution_tag(f[key])


def write_custom_key(save_file, data, key='data'):
    with h5py.File(save_file) as f:
        if key in f:
            print("Overwriting existing key: %s" % key)
            f[key] = data
        else:
            f.create_dataset(key, data=data, compression='gzip', dtype='uint64')
        # add resolution tag
        add_resolution_tag(f[key])


# TODO implement
def write_annotations(save_file, data):
    pass
