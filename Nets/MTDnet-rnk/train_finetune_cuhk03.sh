#!/usr/bin/env sh

.../caffe/caffe-master/build/tools/caffe train \
    --solver=./solver_cuhk03.prototxt --weights=.../bvlc_reference_caffenet.caffemodel
#the model can be found in Offical Caffe website
#Link:http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel