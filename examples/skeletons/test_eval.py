import sys
sys.path.append('../..')


def split_summary(m):
    # split score
    split_score = m.computeSplitScores()
    # skeleton runlen
    resolution = [40., 4., 4.]
    skel_runlens, frag_runlens = m.computeSplitRunlengths(resolution, 1)
    for sk_id, val in skel_runlens.items():
        print("Total runlen of skeleton", sk_id, ":", val, "nm")
        print("(split-score:)", split_score[sk_id])
        for f_id, fval in frag_runlens[sk_id].items():
            print("Segmentation fragment", f_id, "has runlen", fval, "nm")


def explicit_merge_summary(m):
    # merges
    merges = m.computeExplicitMerges()
    if merges:
        for skel_id, labels in merges.items():
            print("Skeleton id", skel_id, "is present in the following segmentation objects:")
            print(labels)
    else:
        print("No false merges")


def heuristic_merge_summary(m):
    resolution = [40., 4., 4.]
    # 1 micrometer is just some dummy value
    max_distance = 1000.
    merges = m.computeHeuristicMerges(resolution, max_distance)
    if merges:
        for skel_id, labels in merges.items():
            print("Skeleton id", skel_id, "is present in the following segmentation objects:")
            print(labels)
    else:
        print("No false merges")


def evaluation_on_testdata(label_file, skeleton_file):
    from cremi_tools.skeletons import build_skeleton_metrics
    m = build_skeleton_metrics(label_file, skeleton_file, 8)
    # split_summary(m)
    # explicit_merge_summary(m)
    heuristic_merge_summary(m)


if __name__ == '__main__':
    label_file = '/home/papec/Work/my_projects/cremi_tools/examples/skeletons/test_ws.n5/watershed'
    skeleton_file = '/home/papec/Work/my_projects/cremi_tools/examples/skeletons/test_skeletons.n5'
    evaluation_on_testdata(label_file, skeleton_file)
