import sys
import vigra

import nifty
import nifty.graph.rag as nrag
sys.path.append('../..')


def segment_block(block_id):
    import cremi_tools.segmentation as cseg
    pmap_path = '' % block_id
    ws_path = '' % block_id

    # load pmap and watersheds
    pmap = vigra.readHDF5(pmap_path, 'data')
    ws = vigra.readHDF5(ws_path, 'data')

    # feature extractor and multicut
    feature_extractor = cseg.MeanBoundaryMapFeatures()
    mc = cseg.Multicut('kernighan-lin')

    # make graph and costs
    rag, edge_probs, _, edge_sizes = feature_extractor(pmap, ws)
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())

    costs = mc.probabilities_to_costs(edge_probs, edge_sizes)
    node_labels = mc(graph, costs)
    return nrag.projectScalarNodeDataToPixels(rag, node_labels)


if __name__ == '__main__':
    save_prefix = ''
    for block_id in range(8):
        seg = segment_block(block_id)
        vigra.writeHDF5(seg, '%s_%i.h5' % (save_prefix, block_id),
                        'data', compression='gzip')
