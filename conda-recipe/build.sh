PY_VER=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))")


# Right now, we only need the segmentation submodule,
# but we should build everythinh at some point.

# Install python modules
mkdir -p ${PREFIX}/cremi_tools
mkdir -p ${PREFIX}/cremi_tools/segmentation

cp cremi_tools/segmentation/*.py ${PREFIX}/cremi_tools/segmentation
cp -r cremi_tools/segmentation ${PREFIX}/cremi_tools/segmentation

echo "${PREFIX}" > ${PREFIX}/lib/python${PY_VER}/site-packages/cremi_tools.pth
python -m compileall ${PREFIX}/cremi_tools
