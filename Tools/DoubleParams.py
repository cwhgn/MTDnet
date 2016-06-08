import scipy.io as sio  
import matplotlib.pyplot as plt  
from PIL import Image,ImageDraw
import numpy as np  
import h5py
import pdb
import os
import sys
caffe_root = '.../caffe/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

def ExtactImageFeature(net,tmppair):
    score = net.predict([tmppair])
    return score

if __name__ == '__main__':

    snetdir = '.../TrainedModel'
    tnetdir = '..../TrainedModel'
    
    # Parameters
    imgw = 227
    imgh = 227
    
    # Load net
    plt.rcParams['figure.figsize'] = (imgh, imgw)
    caffe.set_mode_gpu()
    snet = caffe.Net('.../Tools/deploy4triple.prototxt',snetdir + '/cuhk03_MTDnet.caffemodel',caffe.TEST)
    tnet = caffe.Net('.../Tools/deploy4cross.prototxt', caffe.TEST)

    params = ['conv1','conv2','conv1_p','conv2_p','conv3','conv4','conv5','fc6','fc7','fc8']
    cros_params = ['cros_conv1','cros_conv2','cros_conv1_p','cros_conv2_p','cros_conv3','cros_conv4','cros_conv5','cros_fc6','cros_fc7','cros_fc8']
    print len(params);
    for n in xrange(0,len(params)):
        tnet.params[cros_params[n]][0].data.flat = snet.params[params[n]][0].data.flat
        tnet.params[cros_params[n]][1].data[...] = snet.params[params[n]][1].data
        tnet.params[params[n]][0].data.flat = snet.params[params[n]][0].data.flat
        tnet.params[params[n]][1].data[...] = snet.params[params[n]][1].data

    params = ['feat','feat_p','conv1_n','conv2_n','feat_n']
    for pr in params:
        tnet.params[pr][0].data.flat = snet.params[pr][0].data.flat
        tnet.params[pr][1].data[...] = snet.params[pr][1].data

    tnet.save(tnetdir + '/cuhk03_MTDnet_db.caffemodel')

    ############TEST#############    
    print('exit');
