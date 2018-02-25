import os
import vigra

import nifty.graph.rag as nrag

# TODO use our own metrics impl once they work
from cremi.evaluation import NeuronIds
from cremi import Volume


def eval_block(block_id, res_prefix):
    gt = Volume(vigra.readHDF5('/home/papec/Work/neurodata_hdd/fib25/gt/gt_block%i.h5' % block_id,
                               'data'))
    res = Volume(vigra.readHDF5('%s_%i.h5' % (res_prefix, block_id), 'data'))
    metrics = NeuronIds(gt)
    are = metrics.adapted_rand(res)
    vi_s, vi_m = metrics.voi(res)
    return are, vi_s, vi_m


def eval_all(res_prefix):
    print("Eval for all block")
    for block_id in range(1, 9):
        are, vi_s, vi_m = eval_block(block_id, res_prefix)
        print(vi_s)
        print(vi_m)
        print(are)
        print()


def gt_projection(block_id):
    ws_path = '/home/papec/Work/neurodata_hdd/fib25/watersheds/watershed_block%i.h5' % block_id
    ws = vigra.readHDF5(ws_path, 'data')
    ws = vigra.analysis.labelVolume(ws.astype('uint32'))
    gt = vigra.readHDF5('/home/papec/Work/neurodata_hdd/fib25/gt/gt_block%i.h5' % block_id,
                        'data')

    rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1)
    labeling = nrag.gridRagAccumulateLabels(rag, gt)

    projected = Volume(nrag.projectScalarNodeDataToPixels(rag, labeling))

    metrics = NeuronIds(Volume(gt))
    vi_s, vi_m = metrics.voi(projected)
    are = metrics.adapted_rand(projected)

    print(vi_s)
    print(vi_m)
    print(are)
    print()

    os.remove(ws_path)
    vigra.writeHDF5(ws, ws_path, 'data', compression='gzip')


if __name__ == '__main__':
    # for block_id in range(1, 9):
    #     gt_projection(block_id)
    res_prefix = '/home/papec/Work/neurodata_hdd/fib25/results/res_fullfeats_noweight'
    eval_all(res_prefix)
