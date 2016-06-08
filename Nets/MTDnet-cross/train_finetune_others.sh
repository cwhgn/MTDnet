#!/usr/bin/env sh

.../caffe/caffe-master/build/tools/caffe train \
    --solver=./solver_XX.prototxt --weights=.../cuhk03_MTDnet_db.caffemodel
#the model can be found in "TrainedModel" file, it just doubled the weights to the cross-domain architecture. The detail can be seen in "DoubleParams.py" in "Tools" file.
