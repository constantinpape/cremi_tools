import sys
import vigra

import nifty
import nifty.graph.rag as nrag
sys.path.append('../../..')


def segment_block(block_id, weight_edges=False, cached=False):
    import cremi_tools.segmentation as cseg
    raw_path = '/home/papec/Work/neurodata_hdd/fib25/raw/raw_block%i.h5' % block_id
    pmap_path = '/home/papec/Work/neurodata_hdd/fib25/pmaps/probs_squeezed_block%i.h5' % block_id
    ws_path = '/home/papec/Work/neurodata_hdd/fib25/watersheds/watershed_agglomerated_0.075000_block%i.h5' % block_id

    # load pmap and watersheds
    raw = vigra.readHDF5(raw_path, 'data').astype('float32')
    pmap = vigra.readHDF5(pmap_path, 'data')
    ws = vigra.readHDF5(ws_path, 'data')

    if cached:
        edge_probs = vigra.readHDF5('edge_probs_%i.h5' % block_id, 'data')
        rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1)
        # TODO edge sizes
    else:
        # feature extractor and multicut
        feature_extractor = cseg.RandomForestFeatures('./rf.pkl', True)
        # make graph and costs
        rag, edge_probs, _, edge_sizes = feature_extractor(pmap, ws, raw=raw)
        vigra.writeHDF5(edge_probs, 'edge_probs_%i.h5' % block_id, 'data')
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())

    mc = cseg.Multicut('kernighan-lin', weight_edges=weight_edges)
    if weight_edges:
        costs = mc.probabilities_to_costs(edge_probs, edge_sizes)
    else:
        costs = mc.probabilities_to_costs(edge_probs)
    node_labels = mc(graph, costs)
    return nrag.projectScalarNodeDataToPixels(rag, node_labels)


if __name__ == '__main__':
    save_prefix = '/home/papec/Work/neurodata_hdd/fib25/results/res_fullfeats_noweight_agglomerated'
    for block_id in range(1, 9):
        print("Segmenting block", block_id)
        seg = segment_block(block_id, False, False)
        vigra.writeHDF5(seg, '%s_%i.h5' % (save_prefix, block_id),
                        'data', compression='gzip')
