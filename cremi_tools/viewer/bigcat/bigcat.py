import os
import subprocess


def view(ram_limit=16):
    file_path = os.path.split(os.path.realpath(__file__))[0]
    jar = os.path.join(file_path, 'bigcat-0.0.3-SNAPSHOT-jar-with-dependencies.jar')
    command = ["java", "-Xmx%iG" % ram_limit,
               "-XX:+UseConcMarkSweepGC", "-jar", jar]
    subprocess.call(command)
