import os
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
    block_size_ = ','.join(block_size)
    if output_path is None:
        output_path = input_path
    command = ['java', '-Dspark.master=\'local[*]\'', '-cp',
               'bdv.bigcat.util.HDFConverter',
               '-g', input_path, '-d', input_dataset,
               '-G', output_path, '-b', block_size_,
               output_dataset]
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
    assert os.path.exists(os.path.join(input_path, input_dataset, 's0'))
    assert isinstance(sampling_factors, list)
    assert isinstance(block_sizes, list)
    assert isinstance(m_best, list)
    assert len(block_sizes) <= len(sampling_factors)
    assert len(m_best) <= len(sampling_factors)

    command = ['java', '-Dspark.master=\'local[*]\'', '-cp',
               'bigcat-spark-downsampler-0.0.1-SNAPSHOT-shaded.jar',
               'bdv.bigcat.spark.SparkDownsampler',
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
