import os
import z5py
from subprocess import call


# Example Command:
# java -Dspark.master='local[*]'
# -cp target/hdf-n5-converter-0.0.1-SNAPSHOT-shaded.jar
# bdv.bigcat.util.HDFConverter
# -g /home/papec/mnt/nrs/lauritzen/02/workspace.n5/
# -d filtered/segmentations/multicut_more_features/
# filtered/segmentations/multicut_more_features/s0
# -b 256,256,26
def create_label_multiset(input_path, input_dataset, output_dataset,
                          block_size, output_path=None):
    assert os.path.exists(input_path), input_path
    assert isinstance(block_size, list)
    block_size_ = ','.join(map(str, block_size))
    if output_path is None:
        output_path = input_path

    file_path = os.path.split(os.path.realpath(__file__))[0]
    jar = os.path.join(file_path, 'hdf-n5-converter-0.0.1-SNAPSHOT-shaded.jar')

    f_out = z5py.File(output_path)
    if output_dataset not in f_out:
        f_out.create_group(output_dataset)
    out_ds = '/'.join((output_dataset, 's0'))

    command = ['java', '-Dspark.master=local[*]',
               '-cp', jar, 'bdv.bigcat.util.HDFConverter',
               '-g', input_path, '-d', input_dataset,
               '-G', output_path, '-b', block_size_,
               out_ds]
    call(command)


# Example Command:
# java -Dspark.master='local[*]'
# -cp target/bigcat-spark-downsampler-0.0.1-SNAPSHOT-shaded.jar
# bdv.bigcat.spark.SparkDownsampler
# -g /home/papec/mnt/nrs/lauritzen/02/workspace.n5/
# -d filtered/segmentations/multicut_more_features/
# -b 256,256,25 -b 25,128,128
# -m 5
# 2,2,1 2,2,1 2,2,1 2,2,2
def downsample_labels(input_path, input_dataset, sampling_factors,
                      block_sizes, m_best):
    assert os.path.exists(input_path), input_path
    full_input_path = os.path.join(input_path, input_dataset, 's0')
    assert os.path.exists(full_input_path), full_input_path
    assert isinstance(sampling_factors, list)
    assert isinstance(block_sizes, list)
    assert isinstance(m_best, list)
    assert len(block_sizes) <= len(sampling_factors)
    assert len(m_best) <= len(sampling_factors)

    file_path = os.path.split(os.path.realpath(__file__))[0]
    jar = os.path.join(file_path, 'bigcat-spark-downsampler-0.0.1-SNAPSHOT-shaded.jar')

    command = ['java', '-Dspark.master=local[*]',
               '-cp', jar, 'bdv.bigcat.spark.SparkDownsampler',
               '-g', input_path, '-d', input_dataset]
    for block_size in block_sizes:
        command.append('-b')
        command.append(','.join(map(str, block_size)))
    for m in m_best:
        command.append('-m')
        command.append(str(m))
    for sampling_factor in sampling_factors:
        command.append(','.join(map(str, sampling_factor)))
    call(command)
