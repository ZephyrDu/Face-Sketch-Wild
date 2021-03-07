#!/bin/bash

mkdir $PWD/data
mkdir $PWD/pretrain_model
echo 'Downloading Datasets'
wget http://www.visionlab.cs.hku.hk/data/Face-Sketch-Wild/datasets.tgz -P $PWD/data
echo 'Downloading Pretrain Models'
wget http://www.visionlab.cs.hku.hk/data/Face-Sketch-Wild/models.tgz -P $PWD/pretrain_model


