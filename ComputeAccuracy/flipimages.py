#coding=utf-8

import scipy.io as sio  
import matplotlib.pyplot as plt  
from PIL import Image,ImageDraw
import numpy as np  
import h5py
import pdb
import os

def Crop(img,size):
    margin = (256-size)/2
    box = (margin,margin,256-margin-1,256-margin-1)
    cimg=img.crop(box)
    return cimg

def FlipImages(indir,ids):
    #get total pairs
    pairidx = np.zeros((ids,2))
    for id in xrange(0,ids):
        for i in xrange(0,5):
            tmppath = indir+'/'+'CamA_%04d_%02d.jpg' %(id+1,i+1)
            if os.path.exists(tmppath):
                pairidx[id,0] = pairidx[id,0] + 1
                img = Image.open(tmppath);
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img.save(indir+'/'+'CamA_%04d_%02d_mirror.jpg' %(id+1,i+1))
            else:
                break
        for j in xrange(0,5):
		    #j=j+5;if cuhk03 sets
            tmppath = indir+'/'+'CamB_%04d_%02d.jpg' %(id+1,j+1)
            if os.path.exists(tmppath):
                pairidx[id,1] = pairidx[id,1] + 1
                img = Image.open(tmppath);
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img.save(indir+'/'+'CamB_%04d_%02d_mirror.jpg' %(id+1,j+1))
            else:
                break
    totalimgs = sum(pairidx[:,0])+sum(pairidx[:,1])
    pospairs = sum(pairidx[:,0]*pairidx[:,1])
    print 'Total mirror images is %d and pairs is %d.' %(totalimgs,pospairs)
    return pairidx;

def CropImages(indir,ids,size):
    #get total pairs
    pairidx = np.zeros((ids,2))
    for id in xrange(0,ids):
        for i in xrange(0,5):
            tmppath = indir+'/'+'CamA_%04d_%02d.jpg' %(id+1,i+1)
            if os.path.exists(tmppath):
                pairidx[id,0] = pairidx[id,0] + 1
                img = Image.open(tmppath);
                img = img.resize((256,256), Image.ANTIALIAS)
                img = Crop(img,size);
                img.save(indir+'/'+'CamA_%04d_%02d_crop.jpg' %(id+1,i+1))
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img.save(indir+'/'+'CamA_%04d_%02d_crop_mirror.jpg' %(id+1,i+1))
            else:
                break
        for j in xrange(0,5):
		    #j=j+5;if cuhk03 sets
            tmppath = indir+'/'+'CamB_%04d_%02d.jpg' %(id+1,j+1)
            if os.path.exists(tmppath):
                pairidx[id,1] = pairidx[id,1] + 1
                img = Image.open(tmppath);
                img = img.resize((256,256), Image.ANTIALIAS)
                img = Crop(img,size);
                img.save(indir+'/'+'CamB_%04d_%02d_crop.jpg' %(id+1,j+1))
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img.save(indir+'/'+'CamB_%04d_%02d_crop_mirror.jpg' %(id+1,j+1))
            else:
                break
    totalimgs = sum(pairidx[:,0])+sum(pairidx[:,1])
    pospairs = sum(pairidx[:,0]*pairidx[:,1])
    print 'Total mirror images is %d and pairs is %d.' %(totalimgs,pospairs)
    return pairidx;
    
if __name__ == '__main__':
    
	#The image set wanted to be flipped/cropped.
    indir='.../BuildData/XXData/image/val'#replace val with test0 for cuhk03 dataset
    
	#The ID numbers in the set. This number can be found in "BuildData/ReadMe_BuildData.txt", the size of 1st column of each Valdata_list_XX.npy
    ids=100;
    pidx=FlipImages(indir,ids)
    #pidx=CropImages(indir,ids,227)
    
    ############TEST#############    
    print('exit');
