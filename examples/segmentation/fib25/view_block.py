import sys
import vigra
sys.path.append('../../..')
from cremi_tools.viewer.volumina import view


# pass result list
def view_block(block_id, res_paths, res_labels):
    assert len(res_paths) == len(res_labels)
    pmap_path = '/home/papec/Work/neurodata_hdd/fib25/pmaps/probs_squeezed_block%i.h5' % block_id
    raw_path = '/home/papec/Work/neurodata_hdd/fib25/raw/raw_block%i.h5' % block_id
    ws_path = '/home/papec/Work/neurodata_hdd/fib25/watersheds/watershed_block%i.h5' % block_id
    gt_path = '/home/papec/Work/neurodata_hdd/fib25/gt/gt_block%i.h5' % block_id

    raw = vigra.readHDF5(raw_path, 'data')
    ws = vigra.readHDF5(ws_path, 'data')
    pmap = vigra.readHDF5(pmap_path, 'data')
    gt = vigra.readHDF5(gt_path, 'data')

    results = [vigra.readHDF5(rp, 'data') for rp in res_paths]

    view([raw, ws, pmap, gt] + results,
         ['raw', 'ws', 'pmap', 'gt'] + res_labels)


if __name__ == '__main__':
    res_paths = ['/home/papec/Work/neurodata_hdd/fib25/results/res_fullfeatures_noweight_1.h5']
    res_labels = ['res']
    view_block(1, res_paths, res_labels)
