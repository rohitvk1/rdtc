#!/bin/bash

# Save current working directory
BASEWD=$(pwd)

# CUB Dataset
mkdir -p data/cub
cd data/cub
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar xf CUB_200_2011.tgz
mv CUB_200_2011/* .
rm -r CUB_200_2011
rm CUB_200_2011.tgz
cd $BASEWD
