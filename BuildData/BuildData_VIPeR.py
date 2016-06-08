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
    if not os.path.exists(outdir+'/'+name):
        os.mkdir(outdir+'/'+name)
    for idx in xrange(0,input_data.shape[0]):
        apath = '%s/cam_a/%s.bmp' %(indir,input_data[idx,1])
        if not os.path.exists(apath):
            print '%s does not exist' %apath;
            return;
        ima = Image.open('%s' %apath)
        ima = ima.resize((imgw,imgh), Image.ANTIALIAS)
        ima.save('%s/%s/CamA_%04d_01.jpg' %(outdir,name,idx+1))

        bpath = '%s/cam_b/%s.bmp' %(indir,input_data[idx,2])
        if not os.path.exists(bpath):
            print '%s does not exist' %bpath;
            return;
        ima = Image.open('%s' %bpath)
        ima = ima.resize((imgw,imgh), Image.ANTIALIAS)
        ima.save('%s/%s/CamB_%04d_01.jpg' %(outdir,name,idx+1))


    print 'ids is %d, CamA images is %d, CamB images is %d.' %(input_data.shape[0],input_data.shape[0],input_data.shape[0])

def BuildTripleFile(txtdir,name,ids):
    eachlet = 10#int(total_samples/(2*pairs))+1;
    
    rest = 10
    #build AAB
    aablist = []
    for id in xrange(0,ids):
        tmpneg = [y for y in [np.random.randint(0, ids) for x in range(eachlet+rest)] if (y!=id)]
        if len(tmpneg) < eachlet:
            print 'Warning! Neglist is not enough!'
            return
        for n in xrange(0,eachlet):
            aablist.append(['CamA_%04d_01.jpg' %(id+1),\
                            'CamB_%04d_01.jpg' %(id+1),\
                            'CamB_%04d_01.jpg' %(tmpneg[n]+1)])
    
    #build BBA
    bbalist = []
    for id in xrange(0,ids):
        tmpneg = [y for y in [np.random.randint(0, ids) for x in range(eachlet+rest)] if (y!=id)]
        if len(tmpneg) <eachlet:
            print 'Warning! Neglist is not enough!'
            return
        for n in xrange(0,eachlet):
            bbalist.append(['CamB_%04d_01.jpg' %(id+1),\
                            'CamA_%04d_01.jpg' %(id+1),\
                            'CamA_%04d_01.jpg' %(tmpneg[n]+1)])
    
    #perm and write
    list = np.vstack([aablist,bbalist])
    perm = np.random.permutation(list.shape[0])
    list = list[perm,:]
    
    pos1_list = open(txtdir+'/VIPeR_%s_pos1.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos1_list.write('%s 0\n' %list[x,0])
    pos1_list.close()
    
    pos2_list = open(txtdir+'/VIPeR_%s_pos2.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos2_list.write('%s 1\n' %list[x,1])
    pos2_list.close()
    
    neg_list = open(txtdir+'/VIPeR_%s_neg.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        neg_list.write('%s 0\n' %list[x,2])
    neg_list.close()
    
    print 'total size of %s data is %d' %(name,list.shape[0])

def BuildRankFile(txtdir,name,ids):
    list=[]
    for id in xrange(0,ids):
        if not os.path.exists(outdir+'/'+name+'/'+'CamA_%04d_01.jpg' %(id+1)): 
            print 'No image is find!'
        count = 0;
        for id_p in xrange(0,ids):
            label=0;
            if id_p==id:
                label=1;
            if not os.path.exists(outdir+'/'+name+'/'+'CamB_%04d_01.jpg' %(id_p+1)): 
                print 'No image is find!'
            list.append(['CamA_%04d_01.jpg' %(id+1),'CamB_%04d_01.jpg' %(id_p+1),label]);
            count = count + 1;
        print count;
    
    #perm and write
    list = np.vstack(list)
    #perm = np.random.permutation(list.shape[0])
    #list = list[perm,:]
    
    pos_list = open(txtdir+'/VIPeR_%srank_pos.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos_list.write('%s %d\n' %(list[x,0],int(list[x,2])))
    pos_list.close()
    
    neg_list = open(txtdir+'/VIPeR_%srank_neg.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        neg_list.write('%s %d\n' %(list[x,1],int(list[x,2])))
    neg_list.close()
    
    print 'total size of %s data is %d' %(name,list.shape[0])

def BuildVIPeRmat(indir):
    all_data=[]
    add_flag=0
    ids=0
    for i in xrange(0,900):
        for v in xrange(0,9):
            ang = v*45
            if os.path.exists('%s/cam_a/%03d_%d.bmp' %(indir,i,ang)):
                #find it in cam_b
                for v1 in xrange(0,9):
                    ang1 = v1*45
                    if os.path.exists('%s/cam_b/%03d_%d.bmp' %(indir,i,ang1)):
                        #add in all_data
                        all_data.append((i,'%03d_%d' %(i,ang),'%03d_%d' %(i,ang1)))
                        ids += 1
                        add_flag = 1
                        break
                if (add_flag == 0):
                    print 'Img %03d_%d.bmp found no matching in cam_b!' %(i,ang)
                    return 0
                break
    #all_data has (id_idx,cam_a,cam_b)
    all_data = np.vstack(all_data)
    print 'There are %d ids in VIPeR' %(ids)
    return all_data
            
if __name__ == '__main__':
        
    #four pathes should be set: txtdir,outdir,valdir,indir.

    #output path for the training list and the valuation list
    txtdir = '.../BuildData/VIPeRData'
    #output path for three folders (train,val,test)
    outdir = '.../BuildData/VIPeRData/image'
    #the Valdata_list_XX.npy if used our recorded val data
    valdir ='.../BuildData'
    #the VIPeR source data
    indir='.../VIPeR'
    
    #parameters
    imgw=48
    imgh=128
    if not os.path.exists(txtdir):
        os.mkdir(txtdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    #preprocessor
    all_data = BuildVIPeRmat(indir)
    #val_ratio=0.5
    #(val_data, train_data) = CreateTrainVal(all_data,val_ratio)#Load if produce val data randomly
    val_data = np.load(r'%s/RecordedValdata_VIPeR.npy' %(valdir))#Load if used our recorded val data
    train_data = Preprocessor(all_data,val_data) 

    #save images for each set
    SaveImgdata(train_data,'train'); 
    SaveImgdata(val_data,'val');

    #save list for training and valuation
    BuildTripleFile(txtdir,'train',train_data.shape[0])
    BuildRankFile(txtdir,'val',val_data.shape[0])#valuation for classfication
    BuildTripleFile(txtdir,'val',val_data.shape[0])#valuation for ranking
