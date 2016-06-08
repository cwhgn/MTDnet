import scipy.io as sio  
import matplotlib.pyplot as plt
#%matplotlib inline  
from PIL import Image,ImageDraw
import numpy as np  
import h5py
import pdb
import os
import sys

def ComputeEuclid(feat_a,feat_b):
    feat_diff = feat_a-feat_b
    score = sum(feat_diff*feat_diff)
    return score

def GetRanks(a1_feats,a2_feats,b1_feats,b2_feats,idx,idx2):
    # Load image a feat
    feat_a1 = a1_feats[idx][idx2,:]
    feat_a2 = a2_feats[idx][idx2,:]
    
    tmp_ranks=[]
    for n in xrange(0,cand_num):
        for n2 in xrange(0,b1_feats[n].shape[0]):
            feat_b1 = b1_feats[n][n2,:]
            feat_b2 = b2_feats[n][n2,:]
            
            score1 = ComputeEuclid(feat_a1,feat_b1)
            score2 = ComputeEuclid(feat_a1,feat_b2)
            score3 = ComputeEuclid(feat_a2,feat_b1)
            score4 = ComputeEuclid(feat_a2,feat_b2)
            score = (score1+score2+score3+score4)/4.0
            tmp_ranks.append((n,n2,score))
            #print 'ID %d image %d and ID %d image %d has similar %f' %(idx,idx2,n,n2,score[0,1])
    tmp_ranks=np.vstack(tmp_ranks)
    
    #rank
    idx_sort=np.argsort(tmp_ranks[:,2])
    tmp_ranks=tmp_ranks[idx_sort,:]
    best_rank=-1
    for i in xrange(0,tmp_ranks.shape[0]):
        if(idx==tmp_ranks[i,0]):
            best_rank=i+1
            break
    
    #print 'ID %d image %d best rank is %d.' %(idx,idx2,best_rank)
    return best_rank

if __name__ == '__main__':
    
    indir = '.../ComputeAccuracy/tmp_file'
    #indir = '/home/whchen/caffe/workspace_triplet/tools/tmp_file'
    
    # Parameters
    feat_lg=512
    test_num=100
    cand_num=100#the same as test_num except for PRID dataset
    
    # Preprocessor
    a1_feats = np.load('%s/a1_feats.npy' %indir)
    a2_feats = np.load('%s/a2_feats.npy' %indir)
    b1_feats = np.load('%s/b1_feats.npy' %indir)
    b2_feats = np.load('%s/b2_feats.npy' %indir)

    ranks=[]
    for idx in xrange(0,test_num):
        print '%d done!' %idx;
        for idx2 in xrange(0,a1_feats[idx].shape[0]):
            tmp_rank = GetRanks(a1_feats,a2_feats,b1_feats,b2_feats,idx,idx2)
            ranks.append(tmp_rank)
    ranks=np.vstack(ranks)
    print 'total id is %d and test image is: %d' %(test_num,ranks.shape[0])
    for k in xrange(0,20):
        if(k==0):
            rank_thrd=1
        else:
            rank_thrd=k*5
        count=0
        for i in xrange(0,ranks.shape[0]):
            if(ranks[i]<=rank_thrd):
                count += 1
        accuracy=float(count)/ranks.shape[0]          
        print 'Rank%d accuray=%f' %(rank_thrd,accuracy)
    
    ############TEST#############    
    print('exit');
