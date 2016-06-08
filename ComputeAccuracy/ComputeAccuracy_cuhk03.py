import scipy.io as sio  
import matplotlib.pyplot as plt
#%matplotlib inline  
from PIL import Image,ImageDraw
import numpy as np  
import h5py
import pdb
import os
import sys
caffe_root = '.../caffe/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import copy

def Crop(img,size):
    margin = (256-size)/2
    box = (margin,margin,256-margin-1,256-margin-1)
    cimg=img.crop(box)
    return cimg

def GetRanks(net,indir,pidx,idx,idx2):
    # Load image a
    apath = indir+'/'+'CamA_%04d_%02d.jpg' %(idx+1,idx2+1)
    apath_m = indir+'/'+'CamA_%04d_%02d_mirror.jpg' %(idx+1,idx2+1)
    if not os.path.exists(apath):
        print 'Image CamA_%04d_%02d.jpg is not existed.' %(idx+1,idx2+1)
    if not os.path.exists(apath_m):
        print 'Image CamA_%04d_%02d_mirror.jpg is not existed.' %(idx+1,idx2+1)
    
    tmp_ranks=[]
    for n in xrange(0,test_num):
        for n2 in xrange(0,1):
            bpath = indir+'/'+'CamB_%04d_%02d.jpg' %(n+1,n2+6)
            bpath_m = indir+'/'+'CamB_%04d_%02d_mirror.jpg' %(n+1,n2+6)
            if not os.path.exists(bpath):
                print 'Image CamB_%04d_%02d.jpg is not existed.' %(n+1,n2+6)
            if not os.path.exists(bpath_m):
                print 'Image CamB_%04d_%02d_mirror.jpg is not existed.' %(n+1,n2+6)
            
            score = np.zeros([1,8])
            list1 = (apath,apath,apath_m,apath_m,bpath,bpath,bpath_m,bpath_m)
            list2 = (bpath,bpath_m,bpath,bpath_m,apath,apath_m,apath,apath_m)
            
            for i in xrange(0,8):
                tmp_score = ExtractImageFeature(net,list1[i],list2[i]);
                score[0,i] = tmp_score[0,1]
            
            score_mavg = np.mean(score)
            score_avg = np.mean(score[0,0:4])
            tmp_ranks.append((n,score_mavg,score_avg))
    tmp_ranks=np.vstack(tmp_ranks)
    
    #rank
    idx_sort=np.argsort(-tmp_ranks[:,1])
    tmp_ranks=tmp_ranks[idx_sort,:]
    best_rank1=-1
    for i in xrange(0,tmp_ranks.shape[0]):
        if(idx==tmp_ranks[i,0]):
            best_rank1=i+1
            break
    
    idx_sort=np.argsort(-tmp_ranks[:,2])
    tmp_ranks=tmp_ranks[idx_sort,:]
    best_rank2=-1
    for i in xrange(0,tmp_ranks.shape[0]):
        if(idx==tmp_ranks[i,0]):
            best_rank2=i+1
            break
    
    best_rank = [best_rank1,best_rank2]
    best_rank = np.hstack(best_rank)
    print 'ID %d image %d best rank is (%d,%d).' %(idx,idx2,best_rank1,best_rank2)
    return best_rank

def ExtractImageFeature(net,path1,path2):
    net.blobs['data1'].reshape(1,3,imgh,imgw)
    net.blobs['data1'].data[...] = transformer1.preprocess('data1', caffe.io.load_image(path1))
    net.blobs['data2'].reshape(1,3,imgh,imgw)
    net.blobs['data2'].data[...] = transformer2.preprocess('data2', caffe.io.load_image(path2))
    out = net.forward()
    score = out['prob']
    #score = np.hstack((0,np.random.randint(0,1000)/1000.0)).reshape(1,2)
    return score

def BuildPairFile(indir,ids):
    #get total pairs
    pairidx = np.zeros((ids,2))
    for id in xrange(0,ids):
        for i in xrange(0,5):
            tmppath = indir+'/'+'CamA_%04d_%02d.jpg' %(id+1,i+1)
            if os.path.exists(tmppath):
                pairidx[id,0] = pairidx[id,0] + 1
            else:
                break
        for j in xrange(5,10):
            tmppath = indir+'/'+'CamB_%04d_%02d.jpg' %(id+1,j+1)
            if os.path.exists(tmppath):
                pairidx[id,1] = pairidx[id,1] + 1
            else:
                break
    pospairs = sum(pairidx[:,0]*pairidx[:,1])
    print 'Total name pairs is %d.' %pospairs
    return pairidx;

if __name__ == '__main__':
    
    #The model and deploy path
    netdir = '.../'
    #Image path used for testing
    indir = '.../BuildData/cuhk03Data/image/test0'
    
    # Parameters
    imgw = 227
    imgh = 227
    test_num=100
    
    # Load net
    plt.rcParams['figure.figsize'] = (imgh, imgw)
    caffe.set_mode_gpu()
    #MTDnet,MTDnet-cls and MTDnet-cross share the same deploy.prototxt
    net = caffe.Net(netdir + 'Nets/MTDnet/deploy.prototxt',netdir + 'TrainedModels/XX.caffemodel',caffe.TEST)
    
    #transformer
    transformer1 = caffe.io.Transformer({'data1': net.blobs['data1'].data.shape})
    transformer1.set_transpose('data1', (2,0,1))
    transformer1.set_raw_scale('data1', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer1.set_channel_swap('data1', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    transformer2 = caffe.io.Transformer({'data2': net.blobs['data2'].data.shape})
    transformer2.set_transpose('data2', (2,0,1))
    transformer2.set_raw_scale('data2', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer2.set_channel_swap('data2', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
    #Get Pairidx
    pidx = BuildPairFile(indir,test_num);
    
    ranks=[]
    for idx in xrange(0,test_num):
        for idx2 in xrange(0,int(pidx[idx,0])):
            tmp_rank = GetRanks(net,indir,pidx,idx,idx2)
            ranks.append(tmp_rank)
    ranks=np.vstack(ranks)
    print 'total id is %d and test image is: %d' %(test_num,ranks.shape[0])
    for k in xrange(0,20):
        if(k==0):
            rank_thrd=1
        else:
            rank_thrd=k*5
        count=[0,0]
        for i in xrange(0,ranks.shape[0]):
            for j in xrange(0,2):
                if(ranks[i,j]<=rank_thrd):
                    count[j] += 1
        accuracy=[0,0]
        for i in xrange(0,2):
            accuracy[i]=float(count[i])/ranks.shape[0]          
        print 'Rank%d accuray=(%f,%f)' %(rank_thrd,accuracy[0],accuracy[1])
    
    ############TEST#############    
    print('exit');
