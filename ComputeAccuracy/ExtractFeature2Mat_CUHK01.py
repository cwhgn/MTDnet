#coding=utf-8

import scipy.io as sio  
import matplotlib.pyplot as plt  
from PIL import Image,ImageDraw
import numpy as np  
import h5py
import pdb
import os
import sys
caffe_root = '.../caffe/caffe-master-triplet/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import copy

def GetAnum(labeled,v,id):
    a_num=0
    for i in xrange(0,5):
        if ((labeled[v,0][id,i]).shape[1]!=0):
            a_num = a_num + 1
    return a_num

def GetBnum(labeled,v,id):
    b_num=0
    for i in xrange(5,10):
        if ((labeled[v,0][id,i]).shape[1]!=0):
            b_num = b_num + 1
    return b_num

def ExtractFeat(net,image_path):
    net.blobs['data'].reshape(1,3,imgh,imgw)
    net.blobs['data'].data[...] =  transformer.preprocess('data', caffe.io.load_image(image_path))
    out = net.forward()
    score = out['norm']
    return score

if __name__ == '__main__':
        
    #save extracted features
    outdir = '.../ComputeAccuracy/tmp_file'
    #The model and deploy path
    netdir = '.../'
    #Image path used for testing
    indir = '.../BuildData/CUHK01Data/image/val'
    
    # Parameters
    imgw = 227
    imgh = 227
    test_num=100
    cand_num=100
    
    #parameters
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Load net
    plt.rcParams['figure.figsize'] = (imgh, imgw)
    caffe.set_mode_gpu()
    #MTDnet-XX can be MTDnet-rnk and MTDtrp
    net = caffe.Net(netdir + 'Nets/MTDnet-XX/deploy.prototxt',netdir + 'TrainedModels/XX.caffemodel',caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
    #preprocessor
    a1_feats=[]
    for idx in xrange(0,test_num): 
        tmp_feats=[]
        for idx2 in xrange(0,2):
            if not os.path.exists('%s/CamA_%04d_%02d.jpg' %(indir,idx+1,idx2+1)):
                #print 'Image %s/CamA_%04d_%02d.jpg does not exist!' %(indir,idx+1,idx2+1)
                break
            #get image
            #print 'Image %s/CamA_%04d_%02d.jpg is processing...' %(indir,idx+1,idx2+1)
            ima_path='%s/CamA_%04d_%02d.jpg' %(indir,idx+1,idx2+1)
            #get feat
            extract_feat = ExtractFeat(net,ima_path)
            tmp_feat =  copy.deepcopy(extract_feat)
            #print 'tmp_feat shape is', tmp_feat.shape
            tmp_feats.append(tmp_feat)
        tmp_feats = np.vstack(tmp_feats)
        a1_feats.append(tmp_feats)
        print 'CamA ID%s done!' %idx
    np.save('%s/a1_feats.npy' %outdir, a1_feats)

    a2_feats=[]
    for idx in xrange(0,test_num): 
        tmp_feats=[]
        for idx2 in xrange(0,2):
            if not os.path.exists('%s/CamA_%04d_%02d_mirror.jpg' %(indir,idx+1,idx2+1)):
                #print 'Image %s/CamA_%04d_%02d.jpg does not exist!' %(indir,idx+1,idx2+1)
                break
            #get image
            #print 'Image %s/CamA_%04d_%02d.jpg is processing...' %(indir,idx+1,idx2+1)
            ima_path='%s/CamA_%04d_%02d_mirror.jpg' %(indir,idx+1,idx2+1)
            #get feat
            extract_feat = ExtractFeat(net,ima_path)
            tmp_feat =  copy.deepcopy(extract_feat)
            #print 'tmp_feat shape is', tmp_feat.shape
            tmp_feats.append(tmp_feat)
        tmp_feats = np.vstack(tmp_feats)
        a2_feats.append(tmp_feats)
        print 'CamA ID%s done!' %idx
    np.save('%s/a2_feats.npy' %outdir, a2_feats)

    b1_feats=[]
    for idx in xrange(0,cand_num): 
        tmp_feats=[]
        for idx2 in xrange(0,2):
            if not os.path.exists('%s/CamB_%04d_%02d.jpg' %(indir,idx+1,idx2+1)):
                break
            #get image
            imb_path='%s/CamB_%04d_%02d.jpg' %(indir,idx+1,idx2+1)
            #get feat
            extract_feat = ExtractFeat(net,imb_path)
            tmp_feat =  copy.deepcopy(extract_feat)
            #print 'tmp_feat shape is', tmp_feat.shape
            tmp_feats.append(tmp_feat)
        tmp_feats = np.vstack(tmp_feats)
        b1_feats.append(tmp_feats)
        print 'CamB ID%s done!' %idx
    np.save('%s/b1_feats.npy' %outdir, b1_feats)
    
    b2_feats=[]
    for idx in xrange(0,cand_num): 
        tmp_feats=[]
        for idx2 in xrange(0,2):
            if not os.path.exists('%s/CamB_%04d_%02d_mirror.jpg' %(indir,idx+1,idx2+1)):
                break
            #get image
            imb_path='%s/CamB_%04d_%02d_mirror.jpg' %(indir,idx+1,idx2+1)
            #get feat
            extract_feat = ExtractFeat(net,imb_path)
            tmp_feat =  copy.deepcopy(extract_feat)
            #print 'tmp_feat shape is', tmp_feat.shape
            tmp_feats.append(tmp_feat)
        tmp_feats = np.vstack(tmp_feats)
        b2_feats.append(tmp_feats)
        print 'CamB ID%s done!' %idx
    np.save('%s/b2_feats.npy' %outdir, b2_feats)
    
    ############TEST#############    
    print('exit');
