import numpy as np

import vigra
import nifty.ufd as nufd
import nifty.ground_truth as ngt
import nifty.tools as nt


# TODO implement the masked version
def merge_blocks(overlap_ids, overlaps,
                 overlap_coordinates,
                 overlap_dimension,
                 offsets, ovlp_threshold):
    id_a, id_b = overlap_ids
    ovlp_a, ovlp_b = overlaps
    offset_a, offset_b = offsets[id_a], offsets[id_b]
    assert ovlp_a.shape == ovlp_b.shape, "%s, %s" % (str(ovlp_a.shape), str(ovlp_b.shape))

    ovlp_begin = tuple(coord.start for coord in overlap_coordinates)
    ovlp_end = tuple(coord.stop for coord in overlap_coordinates)

    # find the ids ON the actual block boundary
    ovlp_len = ovlp_a.shape[overlapp_dimension]
    ovlp_dim_begin = ovlp_len // 2 if ovlp_len % 2 == 1 else ovlp_len // 2 - 1
    ovlp_dim_end = ovlp_len // 2 + 1
    boundary = tuple(slice(ovlp_begin[i], ovlp_end[i]) if i != overlapp_dimension else
                     slice(ovlp_dim_begin, ovlp_dim_end) for i in range(3))

    # measure all overlaps
    overlaps_ab = ngt.overlap(ovlp_a, ovlp_b)
    overlaps_ba = ngt.overlap(ovlp_b, ovlp_a)
    node_assignment = []

    # find the ids ON the actual block boundary
    segments_a = np.unique(ovlp_a[boundary])
    segments_b = np.unique(ovlp_b[boundary])

    for seg_a in segments_a:

        ovlp_seg_a, counts_seg_a = overlaps_ab.overlapArraysNormalized(seg_a, sorted=True)
        seg_b = ovlp_seg_a[0]

        ovlp_seg_b, counts_seg_b = overlaps_ba.overlapArraysNormalized(seg_b, sorted=True)
        if ovlp_seg_b[0] != seg_a or seg_b not in segments_b:
            continue

        ovlp_measure = (counts_seg_a[0] + counts_seg_b[0]) / 2.
        if ovlp_measure > ovlp_threshold:
            node_assignment.append([seg_a + offset_a, seg_b + offset_b])

    if node_assignment:
        return np.array(node_assignment, dtype='uint32')
    else:
        return None


# TODO this can be parallelized
def make_new_segmentation(segmentation, blocks,
                          block_coordinates, offsets,
                          node_labeling, coordinate_offset):
    # we will write some parts of the volumes multiple times,
    # but that should be ok, because the ids will agree due to the merging
    for block, block_coord, offset in zip(blocks, block_coordinates, offsets):
        local_begin = tuple(c.start - off for c, off in zip(block_coord, offset))
        local_end = tuple(c.stop - off for c, off in zip(block_coord, offset))
        roi = tuple(slice(beg, end) for beg, end in zip(local_begin, local_end))
        segmentation[roi] = nt.take(node_labeling, block)


def find_overlap_dimensions(overlap_ids, block_coordinats):
    overlap_dimensions = {}
    for id_a, id_b in overlap_ids:
        coords_a, coords_b = block_coordinates[id_a], block_coordinates[id_b]
        center_a = tuple((c.start + c.stop) // 2 for c in coords_a)
        center_b = tuple((c.start + c.stop) // 2 for c in coords_b)
        diff = tuple(abs(ca - cb) for ca, cb in zip(center_a, center_b))
        ovlp_dim = tuple(d for d in diff if d == 0)
        assert len(ovlp_dim) == 1, str(len(ovlp_dim))
        overlap_dimension[(id_a, id_b)] = ovlp_dim[0]
    return overlap_dimensions


def stitch_segmentations_by_overlap(blocks, block_coordinates,
                                    overlap_dict, overlap_coord_dict,
                                    out_path, out_key,
                                    ovlp_threshold=.9):
    # validate all inputs
    assert isinstance(blocks, list)
    assert all(isinstance(block, np.ndarray) for block in blocks)
    assert isinstance(block_coordinates, list)
    assert all(isinstance(block_coord, tuple) for block_coord in block_coordinates)
    assert isinstance(overlap_dict, dict)
    assert isinstance(overlap_coord_dict, dict)

    # find the minimum and maximum coordinates
    # (this assumes that we have a proper block as input)
    begins = np.array([[c.start for c in coord] for coord in block_coordinates],
                      dtype='uint32')
    ends = np.array([[c.stop for c in coord] for coord in block_coordinates],
                    dtype='uint32')
    min_coord = tuple(np.min(begins[:, i]) for i in range(3))
    max_coord = tuple(np.max(ends[:, i]) for i in range(3))
    shape = tuple(ma - mi for mi, ma in zip(min_coord, max_coord))

    # TODO this can be parallelised
    # find the offsets for each block and the max node id
    offsets = [block.max() + 1 for block in blocks]
    last_max_id = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets = np.cumsum(offsets)
    number_of_nodes = offsets[-1] + last_max_id

    overlap_dimensions = find_overlap_dimensions(overlap_dict.keys(), block_coordinats)

    # build the final segmentation
    segmentation = np.zeros(shape, dtype='uint32')

    # TODO this can be parallelised
    # iterate over the overlaps to find the node assignments
    node_assignment = []
    for ovlp_ids, overlaps in overlap_dict.items():
        overlap_coordinates = overlap_coord_dict[ovlp_ids]
        node_assignment.append(merge_blocks(ovlp_ids, overlaps,
                                            overlap_coordinates,
                                            overlap_dimensions
                                            offsets, ovlp_threshold))
    node_assignment = np.concatenate([na for na in node_assignment if na is not None],
                                     axis=0)

    # merge nodes with union find
    ufd = nufd.ufd(number_of_nodes)
    ufd.merge(node_assignment)
    node_labeling = ufd.elementLabeling()

    # assign the new ids
    make_new_segmentation(segmentation, blocks, block_coordinates,
                          offsets, node_labeling)
    vigra.analysis.relabelConsecutive(segmentation, out=segmentation)

    # save the new segmentation
    vigra.writeHDF5(segmentation, out_path, out_key, compression='gzip')
