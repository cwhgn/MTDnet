#!/usr/bin/env sh

.../caffe/caffe-master/build/tools/caffe train \
    --solver=./solver_cuhk03.prototxt --weights=.../bvlc_conv12shared.caffemodel
#the model can be found in "TrainedModel" file, it just extracted the weights of the first two convoluational layers in ImageNet model. The detail can be seen in "ExtractFromImageNet.py" in "Tools" file.
