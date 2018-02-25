import sys
# import numpy as np
import vigra

import nifty
import nifty.graph.rag as nrag

# TODO use our own metrics impl once they work
from cremi.evaluation import NeuronIds
from cremi import Volume

sys.path.append('../../..')
import cremi_tools.segmentation as cseg


def agglomerate_sp(ws_path, prob_path, out_path, threshold):
    probs = vigra.readHDF5(prob_path, 'data')

    ws = vigra.readHDF5(ws_path, 'data')
    n_nodes = int(ws.max()) + 1

    rag = nrag.gridRag(ws, numberOfLabels=n_nodes)
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(rag.uvIds())

    agglomerator = cseg.MalaClustering(threshold)
    node_labeling = agglomerator(graph, probs)
    vigra.analysis.relabelConsecutive(node_labeling, out=node_labeling)
    seg = nrag.projectScalarNodeDataToPixels(rag, node_labeling)
    vigra.writeHDF5(seg, out_path, 'data', compression='gzip')


def agglomerate_block(block_id, threshold):
    ws_path = '/home/papec/Work/neurodata_hdd/fib25/watersheds/watershed_block%i.h5' % block_id
    prob_path = './edge_probs_%i.h5' % block_id
    out_path = '/home/papec/Work/neurodata_hdd/fib25/watersheds/watershed_agglomerated_%f_block%i.h5' % \
        (threshold, block_id)
    agglomerate_sp(ws_path, prob_path, out_path, threshold)


def agglomerate_sp_eval(ws_path, gt_path, prob_path):

    probs = vigra.readHDF5(prob_path, 'data')

    ws = vigra.readHDF5(ws_path, 'data')
    n_nodes = int(ws.max()) + 1

    rag = nrag.gridRag(ws, numberOfLabels=n_nodes)
    # _, node_sizes = np.unique(ws, return_counts=True)
    # edge_sizes = nrag.accumulateEdgeMeanAndLength(rag, np.zeros(rag.shape, dtype='float32'))[:, 1]
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(rag.uvIds())

    gt = Volume(vigra.readHDF5(gt_path, 'data'))

    # node_factor = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1][::-1]
    node_factor = [.025, .05, .075, .1, .15, .2, .25, .4, .5]

    for nf in node_factor:
        # FIXME agglomerative clustering segfaults
        # n_target_nodes = int(nf * n_nodes)
        # agglomerator = cseg.AgglomerativeClustering(n_target_nodes)
        # node_labeling = agglomerator(graph, probs, edge_sizes=edge_sizes, node_sizes=node_sizes)

        agglomerator = cseg.MalaClustering(nf)
        node_labeling = agglomerator(graph, probs)
        vigra.analysis.relabelConsecutive(node_labeling, out=node_labeling)

        seg = nrag.projectScalarNodeDataToPixels(rag, node_labeling)
        seg = Volume(seg)
        metrics = NeuronIds(gt)
        vi_s, vi_m = metrics.voi(seg)
        are = metrics.adapted_rand(seg)
        print("Evaluation for reduction", nf)
        print("Voi - Split ", vi_s)
        print("Voi - Merge ", vi_m)
        print("Adapted Rand", are)
        print("N-Nodes:", int(node_labeling.max() + 1), '/', n_nodes)


def eval_agglomerate_sp_block(block_id):
    ws_path = '/home/papec/Work/neurodata_hdd/fib25/watersheds/watershed_block%i.h5' % block_id
    gt_path = '/home/papec/Work/neurodata_hdd/fib25/gt/gt_block%i.h5' % block_id
    prob_path = './edge_probs_%i.h5' % block_id
    agglomerate_sp_eval(ws_path, gt_path, prob_path)


if __name__ == '__main__':
    # eval_agglomerate_sp_block(3)
    threshold = 0.075
    for block_id in range(1, 9):
        agglomerate_block(block_id, threshold)
