import os
import nifty.skeletons as nskel


def build_skeleton_metrics(label_file, skeleton_file, n_threads=-1):
    assert os.path.exists(label_file), label_file
    assert os.path.exists(skeleton_file), skeleton_file
    skeleton_ids = os.listdir(skeleton_file)
    skeleton_ids = [int(sk) for sk in skeleton_ids if sk.isdigit()]
    skeleton_ids.sort()
    return nskel.SkeletonMetrics(label_file, skeleton_file, skeleton_ids, n_threads)


def load_skeleton_metrics(label_file, skeleton_file, serialization_file):
    assert os.path.exists(label_file), label_file
    assert os.path.exists(skeleton_file), skeleton_file
    assert os.path.exists(serialization_file), serialization_file
    skeleton_ids = os.listdir(skeleton_file)
    skeleton_ids = [int(sk) for sk in skeleton_ids if sk.isdigit()]
    skeleton_ids.sort()
    return nskel.SkeletonMetrics(label_file, skeleton_file, skeleton_ids, serialization_file)
