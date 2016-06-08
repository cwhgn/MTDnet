#coding=utf-8

import scipy.io as sio  
import matplotlib.pyplot as plt  
from PIL import Image,ImageDraw
import numpy as np  
import h5py
import pdb
import os
import random

def BuildPairFile(indir,name,ids,ratio):
    #get total pairs
    pairidx = np.zeros((ids,2))
    for id in xrange(0,ids):
        for i in xrange(0,5):
            tmppath = indir+'/'+name+'/'+'CamA_%04d_%02d.jpg' %(id+1,i+1)
            if os.path.exists(tmppath):
                pairidx[id,0] = pairidx[id,0] + 1
            else:
                break
        for j in xrange(5,10):
            tmppath = indir+'/'+name+'/'+'CamB_%04d_%02d.jpg' %(id+1,j+1)
            if os.path.exists(tmppath):
                pairidx[id,1] = pairidx[id,1] + 1
            else:
                break
    pospairs = sum(pairidx[:,0]*pairidx[:,1])
    print 'Total pos pairs is %d.' %pospairs
    
    list1=[]
    for id in xrange(0,ids):
        for i in xrange(0,int(pairidx[id,0])):
            for j in xrange(0,int(pairidx[id,1])):
                if not os.path.exists(indir+'/'+name+'/'+'CamA_%04d_%02d.jpg' %(id+1,i+1)): 
                    print 'No right image existed!'
                if not os.path.exists(indir+'/'+name+'/'+'CamB_%04d_%02d.jpg' %(id+1,j+6)): 
                    print 'No right image existed!'
                list1.append(['CamA_%04d_%02d.jpg' %(id+1,i+1),'CamB_%04d_%02d.jpg' %(id+1,j+6),1]);
            tmpneg = [x for x in random.sample(range(ids),int(pairidx[id,1])*ratio+10) if (x!=id and pairidx[x,0]!=0 and pairidx[x,1]!=0)]
            for n in xrange(0,int(pairidx[id,1])*ratio):
                if not os.path.exists(indir+'/'+name+'/'+'CamB_%04d_%02d.jpg' %(tmpneg[n]+1,np.random.randint(0,int(pairidx[tmpneg[n],1]))+6)): 
                    print 'No right image existed!'
                list1.append(['CamA_%04d_%02d.jpg' %(id+1,i+1),\
                             'CamB_%04d_%02d.jpg' %(tmpneg[n]+1,np.random.randint(0,int(pairidx[tmpneg[n],1]))+6),0]);
    list2=[]
    for id in xrange(0,ids):
        for i in xrange(0,int(pairidx[id,1])):
            for j in xrange(0,int(pairidx[id,0])):
                if not os.path.exists(indir+'/'+name+'/'+'CamB_%04d_%02d.jpg' %(id+1,i+6)): 
                    print 'No right image existed!'
                if not os.path.exists(indir+'/'+name+'/'+'CamA_%04d_%02d.jpg' %(id+1,j+1)): 
                    print 'No right image existed!'
                list2.append(['CamB_%04d_%02d.jpg' %(id+1,i+6),'CamA_%04d_%02d.jpg' %(id+1,j+1),1]);
            tmpneg = [x for x in random.sample(range(ids),int(pairidx[id,0])*ratio+10) if (x!=id and pairidx[x,0]!=0 and pairidx[x,1]!=0)]
            for n in xrange(0,int(pairidx[id,0])*ratio):
                if not os.path.exists(indir+'/'+name+'/'+'CamA_%04d_%02d.jpg' %(tmpneg[n]+1,np.random.randint(0,int(pairidx[tmpneg[n],0]))+1)): 
                    print 'No right image existed!'
                list2.append(['CamB_%04d_%02d.jpg' %(id+1,i+6),\
                             'CamA_%04d_%02d.jpg' %(tmpneg[n]+1,np.random.randint(0,int(pairidx[tmpneg[n],0]))+1),0]);
    
    #perm and write
    list = np.vstack([list1,list2])
    perm = np.random.permutation(list.shape[0])
    list = list[perm,:]
    
    pos_list = open(outdir+'/cuhk03_%s4cross_pos.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos_list.write('%s %d\n' %(list[x,0],int(list[x,2])))
    pos_list.close()
    
    neg_list = open(outdir+'/cuhk03_%s4cross_neg.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        neg_list.write('%s %d\n' %(list[x,1],1-int(list[x,2])))
    neg_list.close()
    
    print 'total size of %s data is %d' %(name,list.shape[0])
            
if __name__ == '__main__':
        
    outdir = '.../BuildData/cuhk03Data'
    indir='.../BuildData/cuhk03Data/image'
    
    #parameters
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    neg_ratio=1#neg ratio to pos
    
    #preprocessor
    BuildPairFile(indir,'train',1160,neg_ratio)
    
    ############TEST#############    
    print('exit');
