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

    snetdir = '.../caffe/models/bvlc_reference_caffenet'
    tnetdir = '..../TrainedModel'
    
    # Parameters
    imgw = 227
    imgh = 227
    
    # Load net
    plt.rcParams['figure.figsize'] = (imgh, imgw)
    caffe.set_mode_gpu()
    snet = caffe.Net(snetdir + '/deploy.prototxt',snetdir + '/bvlc_reference_caffenet.caffemodel',caffe.TEST)
    tnet = caffe.Net('.../Tools/deploy4triple.prototxt', caffe.TEST)

    params = ['conv1','conv2','conv1_p','conv2_p']
    for pr in params:
        tnet.params[pr][0].data.flat = snet.params[pr][0].data.flat
        tnet.params[pr][1].data[...] = snet.params[pr][1].data

    tnet.save(tnetdir + '/bvlc_conv12shared.caffemodel')

    ############TEST#############    
    print('exit');
