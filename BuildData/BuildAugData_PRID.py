#coding=utf-8

import scipy.io as sio  
import matplotlib.pyplot as plt  
from PIL import Image,ImageDraw
import numpy as np  
import h5py
import pdb
import os

def Isval(val_data,n):
    val_flag=0
    for i in xrange(0,val_data.shape[0]):
        if (int(val_data[i,0])==n):
            val_flag=1
            break
    return val_flag

def Preprocessor(all_data,val_data):
    train_data=[]
    trainpairs=0
    for i in xrange(0,all_data.shape[0]):
        if (Isval(val_data,int(all_data[i,0]))==0):
            train_data.append(all_data[i,:])
            trainpairs += 1
    train_data = np.vstack(train_data)
    print 'trainpairs=%d' %(trainpairs)
    return train_data

def BuildTripleFile(name,ids):
    #build AAB
    aablist = []
    for id in xrange(0,ids):
        for n in xrange(0,ids):
            if id!=n:
                aablist.append(['CamA_%04d_01.jpg' %(id+1),\
                                'CamB_%04d_01.jpg' %(id+1),\
                                'CamB_%04d_01.jpg' %(n+1)])
    
    #build BBA
    bbalist = []
    for id in xrange(0,ids):
        for n in xrange(0,ids):
            if id!=n:
                bbalist.append(['CamB_%04d_01.jpg' %(id+1),\
                                'CamA_%04d_01.jpg' %(id+1),\
                                'CamA_%04d_01.jpg' %(n+1)])
    
    #perm and write
    list = np.vstack([aablist,bbalist])
    perm = np.random.permutation(list.shape[0])
    list = list[perm,:]

    return list

def WriteList(list,txtdir,name):      
    pos1_list = open(txtdir+'/PRID_%s_pos1.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos1_list.write('%s 0\n' %list[x,0])
    pos1_list.close()
    
    pos2_list = open(txtdir+'/PRID_%s_pos2.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos2_list.write('%s 1\n' %list[x,1])
    pos2_list.close()
    
    neg_list = open(txtdir+'/PRID_%s_neg.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        neg_list.write('%s 0\n' %list[x,2])
    neg_list.close()
    
    print 'total size of %s data is %d' %(name,list.shape[0])

def BuildCUHK03TripleFile(indir,name,ids):
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
    pairs = sum(pairidx[:,0]*pairidx[:,1])
    eachlet = 1#int(total_samples/(2*pairs))+1;

    rest = 10
    #build AAB
    aablist = []
    for id in xrange(0,ids):
        for i in xrange(0,int(pairidx[id,0])):
            for j in xrange(5,int(pairidx[id,1])+5):
                tmpneg = [y for y in [np.random.randint(0, ids) for x in range(eachlet+rest)] if (y!=id)]
                if len(tmpneg) < eachlet:
                    print 'Warning! Neglist is not enough!'
                    return
                for n in xrange(0,eachlet):
                    aablist.append(['CamA_%04d_%02d.jpg' %(id+1,i+1),\
                                    'CamB_%04d_%02d.jpg' %(id+1,j+1),\
                                    'CamB_%04d_%02d.jpg' %(tmpneg[n]+1,np.random.randint(0,int(pairidx[tmpneg[n],1]))+6)])

    #build BBA
    bbalist = []
    for id in xrange(0,ids):
        for j in xrange(5,int(pairidx[id,1])+5):
            for i in xrange(0,int(pairidx[id,0])):
                tmpneg = [y for y in [np.random.randint(0, ids) for x in range(eachlet+rest)] if (y!=id and y!=1152)]
                if len(tmpneg) <eachlet:
                    print 'Warning! Neglist is not enough!'
                    return
                for n in xrange(0,eachlet):
                    bbalist.append(['CamB_%04d_%02d.jpg' %(id+1,j+1),\
                                    'CamA_%04d_%02d.jpg' %(id+1,i+1),\
                                    'CamA_%04d_%02d.jpg' %(tmpneg[n]+1,np.random.randint(0,int(pairidx[tmpneg[n],0]))+1)])

    #perm and write
    list = np.vstack([aablist,bbalist])
    perm = np.random.permutation(list.shape[0])
    list = list[perm,:]
    
    return list

def Addpath(path,list):
    newlist=[]
    for i in xrange(0,list.shape[0]):
        tmplist=[]
        for j in xrange(0,3):
            tmplist.append(path+list[i,j])
        tmplist = np.hstack(tmplist)
        newlist.append(tmplist)
    newlist = np.vstack(newlist)
    print 'list is',newlist.shape
    print 'An example is',newlist[10]
    return newlist

if __name__ == '__main__':
        
    #five pathes should be set: txtdir,valdir,s_indir,t_indir,pb_indir.

    #output path for the training list and the valuation list
    txtdir = '.../BuildData/PRIDData'
    #the Valdata_list_PRID.npy if used our recorded val data
    valdir ='.../BuildData'
    #Relative image path of PRID
    t_indir = 'PRIDData/image'
    #Relative image path of cuhk03
    s_indir = 'cuhk03Data/image' 
    #The public path where PRIDData and cuhk03Data are saved.
    pb_indir = '.../BuildData'  
    
    #parameters
    imgw=64
    imgh=128
    if not os.path.exists(txtdir):
        os.mkdir(txtdir)
    
    #preprocessor
    all_data = np.arange(1, 201, 1).reshape(200,1)
    val_data = np.load(r'%s/RecordedValdata_PRID.npy' %(txtdir)) 
    train_data = Preprocessor(all_data,val_data)  
    t_list = BuildTripleFile('train',train_data.shape[0])
    s_list = BuildCUHK03TripleFile(pb_indir+'/'+s_indir,'train',1160)

    #perm and write
    t_list = Addpath(t_indir+'/train/',t_list)
    s_list = Addpath(s_indir+'/train/',s_list)
    list = np.vstack([t_list,s_list[0:t_list.shape[0],:]])
    perm = np.random.permutation(list.shape[0])
    list = list[perm,:]

    WriteList(list,txtdir,'trainaug');
