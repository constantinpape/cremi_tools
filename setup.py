from distutils.core import setup
from cremi_tools import __version__


setup(name='cremi_tools',
      version=__version__,
      description='Tools for cremi challenge and neuron segmentation',
      author='Constantin Pape',
      packages=['cremi_tools',
                'cremi_tools/metrics',
                'cremi_tools/viewer/volumina',
                'cremi_tools/alignment',
                'cremi_tools/viewer',
                'cremi_tools/segmentation'],
      scripts=['scripts/view_container'])
