#! /bin/sh

mkdir -p data.mb/unzip
cd data.mb/unzip

# 2014 dataset
wget -r -np -A png,pfm,txt -X "/stereo/data/scenes2014/datasets/*-perfect/" http://vision.middlebury.edu/stereo/data/scenes2014/datasets/

# 2006 dataset
wget -r -np -A png,txt http://vision.middlebury.edu/stereo/data/scenes2006/HalfSize/

# 2005 dataset
wget -r -np -A png,txt http://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/

# 2003 dataset
mkdir vision.middlebury.edu/stereo/data/scenes2003/
pushd .
cd vision.middlebury.edu/stereo/data/scenes2003/
wget http://vision.middlebury.edu/stereo/data/scenes2003/newdata/full/conesH-ppm-2.zip
wget http://vision.middlebury.edu/stereo/data/scenes2003/newdata/full/teddyH-ppm-2.zip
unzip conesH-ppm-2.zip
unzip teddyH-ppm-2.zip
popd

# 2001 dataset
wget -r -np -A pgm,ppm,txt http://vision.middlebury.edu/stereo/data/scenes2001/data/
# get tsukuba nonocc mask
pushd .
cd vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba
wget http://vision.middlebury.edu/stereo/eval/newEval/tsukuba/nonocc.png
popd

# eval3 train/test set
wget http://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip
unzip MiddEval3-data-H.zip

