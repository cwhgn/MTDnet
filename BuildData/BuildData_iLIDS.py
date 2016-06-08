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

def CreateTrainVal(all_data,val_ratio):
    val_num=int(all_data.shape[0]*val_ratio)
    perm = np.random.permutation(all_data.shape[0])
    data = all_data[perm,]
    val_data = data[0:val_num,]
    train_data = data[val_num:all_data.shape[0],]
    print "Successfully create val data! Val data is ",val_data.shape
    print "Successfully create train data! Train data is ",train_data.shape
    #save test data
    np.save('%s/val_data.npy' %(outdir), val_data)
    print "test data is saved!"
    return (val_data,train_data)

def SaveImgdata(input_data,name):
    total=0
    if not os.path.exists(outdir+'/'+name):
        os.mkdir(outdir+'/'+name)
    for idx in xrange(0,input_data.shape[0]):
        total = total + input_data[idx,1];
        for idx2 in xrange(0,input_data[idx,1]):
            apath = '%s/%04d%03d.jpg' %(indir,input_data[idx,0],idx2+1)
            if not os.path.exists(apath):
                print '%s does not exist' %apath;
                return;
            ima = Image.open('%s' %apath)
            ima = ima.resize((imgw,imgh), Image.ANTIALIAS)
            ima.save('%s/%s/%04d_%02d.jpg' %(outdir,name,idx+1,idx2+1))

    print 'ids is %d, total images is %d.' %(input_data.shape[0],total)

def BuildTripleFile(txtdir,name,ids):
    #get total pairs
    pairidx = np.zeros((ids,1))
    for id in xrange(0,ids):
        for i in xrange(0,100):
            tmppath = outdir +'/'+name+'/'+'%04d_%02d.jpg' %(id+1,i+1)
            if os.path.exists(tmppath):
                pairidx[id,0] = pairidx[id,0] + 1
            else:
                break

    eachlet = 10;
    
    rest = 5
    #build all
    alllist = []
    for id in xrange(0,ids):
        for i in xrange(0,int(pairidx[id,0])):
            for j in xrange(0,int(pairidx[id,0])):
                if i==j:
                    continue;
                tmpneg = [y for y in [np.random.randint(0, ids) for x in range(eachlet+rest)] if (y!=id)]
                if len(tmpneg) < eachlet:
                    print 'Warning! Neglist is not enough!'
                    return
                for n in xrange(0,eachlet):
                    alllist.append(['%04d_%02d.jpg' %(id+1,i+1),\
                                    '%04d_%02d.jpg' %(id+1,j+1),\
                                    '%04d_%02d.jpg' %(tmpneg[n]+1,np.random.randint(0,int(pairidx[tmpneg[n],0]))+1)])
    
    #perm and write
    list = np.vstack(alllist)
    perm = np.random.permutation(list.shape[0])
    list = list[perm,:]
    
    pos1_list = open(txtdir+'/iLIDS_%s_pos1.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos1_list.write('%s 0\n' %list[x,0])
    pos1_list.close()
    
    pos2_list = open(txtdir+'/iLIDS_%s_pos2.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos2_list.write('%s 1\n' %list[x,1])
    pos2_list.close()
    
    neg_list = open(txtdir+'/iLIDS_%s_neg.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        neg_list.write('%s 0\n' %list[x,2])
    neg_list.close()
    
    print 'total size of %s data is %d' %(name,list.shape[0])

def BuildRankFile(txtdir,name,ids):
    #get total pairs
    pairidx = np.zeros((ids,1))
    for id in xrange(0,ids):
        for i in xrange(0,100):
            tmppath = outdir+'/'+name+'/'+'%04d_%02d.jpg' %(id+1,i+1)
            if os.path.exists(tmppath):
                pairidx[id,0] = pairidx[id,0] + 1
            else:
                break

    list=[]
    allcount=0
    for id in xrange(0,ids):
        for i in xrange(1,int(pairidx[id,0])):
            if not os.path.exists(outdir+'/'+name+'/'+'%04d_%02d.jpg' %(id+1,i+1)): 
                print 'No image is find!'
            count = 0;
            for id_p in xrange(0,ids):
                label=0;
                if id_p==id:
                    label=1;
                if not os.path.exists(outdir+'/'+name+'/'+'%04d_01.jpg' %(id_p+1)): 
                    print 'No image is find!'
                list.append(['%04d_%02d.jpg' %(id+1,i+1),'%04d_01.jpg' %(id_p+1),label]);
                count = count + 1;
            allcount = allcount + count
            print count;
    
    #perm and write
    list = np.vstack(list)
    #perm = np.random.permutation(list.shape[0])
    #list = list[perm,:]
    
    pos_list = open(txtdir+'/iLIDS_%srank_pos.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos_list.write('%s %d\n' %(list[x,0],int(list[x,2])))
    pos_list.close()
    
    neg_list = open(txtdir+'/iLIDS_%srank_neg.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        neg_list.write('%s %d\n' %(list[x,1],int(list[x,2])))
    neg_list.close()
    
    print 'total size of %s data is %d' %(name,list.shape[0])

def BuildiLIDSmat(indir):
    all_data=[]
    add_flag=0
    ids=0
    for i in xrange(0,120):
        count=0
        for v in xrange(0,8):
            if os.path.exists('%s/%04d%03d.jpg' %(indir,i,v+1)):
                count += 1
                #find it in cam_b
            else:
                break
        if count>0 :
            all_data.append((i,count))
            ids += count
    #all_data has (id_idx,img_num)
    all_data = np.vstack(all_data)
    print 'There are %d ids in iLIDS' %(ids)
    return all_data
            
if __name__ == '__main__':
        
    #four pathes should be set: txtdir,outdir,valdir,indir.

    #output path for the training list and the valuation list
    txtdir = '.../BuildData/iLIDSData'
    #output path for three folders (train,val,test)
    outdir = '.../BuildData/iLIDSData/image'
    #the Valdata_list_XX.npy if used our recorded val data
    valdir ='.../BuildData'
    #the iLIDS source data
    indir='.../iLIDS'
    
    #parameters 
    imgw=227
    imgh=227
    if not os.path.exists(txtdir):
        os.mkdir(txtdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    #preprocessor
    all_data = BuildiLIDSmat(indir)
    #val_ratio=0.5
    #(val_data, train_data) = CreateTrainVal(all_data,val_ratio)#Load if produce val data randomly
    val_data = np.load(r'%s/RecordedValdata_iLIDS.npy' %(valdir))#Load if used our recorded val data
    train_data = Preprocessor(all_data,val_data) 

    #save images for each set
    SaveImgdata(train_data,'train'); 
    SaveImgdata(val_data,'val');

    #save list for training and valuation
    BuildTripleFile(txtdir,'train',train_data.shape[0])
    BuildRankFile(txtdir,'val',val_data.shape[0])#valuation for classfication
    BuildTripleFile(txtdir,'val',val_data.shape[0])#valuation for ranking
