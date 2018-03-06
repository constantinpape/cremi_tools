import numpy as np
import sys
sys.path.append('../..')


def eval_test(label_file, skeleton_file):
    from cremi_tools.skeletons import SkeletonMetrics
    m = SkeletonMetrics(label_file, skeleton_file, 8)

    # split score
    # split_score = m.compute_split_score()
    # for sk_id, val in split_score.items():
    #     print(sk_id, val)

    # skeleton runlen
    # resolution = np.array([40, 4, 4])
    # skel_runlens, frag_runlens = m.compute_split_runlength(resolution)
    # for sk_id, val in skel_runlens.items():
    #     print("Total runlen of skeleton", sk_id, ":", val, "nm")
    #     print("(split-score:)", split_score[sk_id])
    #     for f_id, fval in frag_runlens[sk_id].items():
    #         print("Segmentation fragment", f_id, "has runlen", fval, "nm")

    # merges
    merges = m.compute_merges()
    if merges:
        for skel_id, labels in merges.items():
            print("Skeleton id", skel_id, "is present in the following segmentation objects:")
            print(labels)
    else:
        print("No false merges")


if __name__ == '__main__':
    label_file = '/home/papec/Work/my_projects/cremi_tools/examples/skeletons/test_ws.n5/watershed'
    skeleton_file = '/home/papec/Work/my_projects/cremi_tools/examples/skeletons/test_skeletons.n5'
    eval_test(label_file, skeleton_file)
