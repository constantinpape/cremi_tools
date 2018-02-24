import vigra

# TODO use our own metrics impl once they work
from cremi.evaluation import NeuronIds
from cremi import Volume


def eval_block(block_id, res_prefix):
    gt = Volume(vigra.readHDF5('/home/constantin/Work/neurodata_hdd/FIB25/fib_25_blocks/gt/gt_block%i.h5' % block_id,
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


if __name__ == '__main__':
    res_prefix = '/home/constantin/Work/neurodata_hdd/FIB25/fib_25_blocks/results/res_fullfeatures_noweight'
    eval_all(res_prefix)
