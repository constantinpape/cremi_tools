import os
import subprocess
from itertools import chain


def view(paths, paths_in_file, ranges, resolution=[1, 1, 1]):
    assert len(paths) == len(paths_in_file)
    assert len(paths) == len(ranges)
    assert all(os.path.exists(p) for p in paths)
    assert all(len(r) == 2 for r in ranges)

    file_path = os.path.split(os.path.realpath(__file__))[0]
    jar = os.path.join(file_path, 'simple-viewer-0.0.1-SNAPSHOT.jar')
    command = ["java", "-jar", jar]

    inputs = [["-i", p] for p in paths]
    inputs = list(chain(*inputs))
    command.extend(inputs)

    keys = [["-d", k] for k in paths_in_file]
    keys = list(chain(*keys))
    command.extend(keys)

    input_ranges = [["-c", ",".join(map(str, r))] for r in ranges]
    input_ranges = list(chain(*input_ranges))
    command.extend(input_ranges)

    command.extend(["-r", ",".join(map(str, resolution))])

    # for c in command:
    #     print(c)

    subprocess.call(command)
