#coding=utf-8

import scipy.io as sio  
import matplotlib.pyplot as plt  
from PIL import Image,ImageDraw
import numpy as np  
import h5py
import pdb
import os

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

def Isval(val_data,v,id):
    val_flag=0
    for t in xrange(0,val_data.shape[0]):
        if (val_data[t,0]==v+1 and val_data[t,1]==id+1):
            val_flag=1
            break
    return val_flag
    
def Istest(test_data,v,id):
    test_flag=0
    for t in xrange(0,test_data.shape[0]):
        if (test_data[t,0]==v+1 and test_data[t,1]==id+1):
            test_flag=1
            break
    return test_flag

def Istrain(train_data,v,id):
    train_flag=0
    for t in xrange(0,train_data.shape[0]):
        if (train_data[t,0]==v+1 and train_data[t,1]==id+1):
            train_flag=1
            break
    return train_flag

def Preprocessor(labeled,test_data,val_data):
    train_data=[]
    cam_ratio=[]
    ids_tnum=0
    img_tnum=0
    pair_tnum=0
    ids_num=0
    img_num=0
    pair_num=0
    for v in xrange(start_cam,end_cam):
        ids = labeled[v,0].shape[0]
        ids_num = ids_num + ids
        for id in xrange(0,ids):
            a_num=GetAnum(labeled,v,id)
            b_num=GetBnum(labeled,v,id)
            img_num = img_num + a_num + b_num
            pair_num = pair_num + a_num * b_num
            if (Istest(test_data,v, id)==0 and Isval(val_data,v, id)==0):
                ids_tnum = ids_tnum + 1
                img_tnum = img_tnum + a_num + b_num
                pair_tnum = pair_tnum + a_num * b_num
                train_data.append((v+1,id+1))
        if(v-start_cam>0):
            cam_ratio.append(ids_tnum-np.hstack(cam_ratio).sum())
        else:
            cam_ratio.append(ids_tnum)
    cam_ratio=np.hstack(cam_ratio)
    train_data=np.vstack(train_data)
    #print "test_data:",test_data
    #print "val_data:",val_data
    #print "train_data:",train_data
    print "cam_ratio:",cam_ratio
    print "train: ids_tnum:%d,img_tnum=%d,pair_tnum=%d" %(ids_tnum,img_tnum,pair_tnum)
    print "total: ids_num:%d,img_num=%d,pair_num=%d" %(ids_num,img_num,pair_num)
    return train_data

def CreateValdata(labeled,test_data):
    val_num=100
    val_data = []
    sum_data = []
    for v in xrange(start_cam,end_cam):
        for id in xrange(0,labeled[v,0].shape[0]):
            sum_data.append((v+1,id+1))
    sum_data = np.vstack(sum_data)
    perm = np.random.permutation(sum_data.shape[0])
    sum_data = sum_data[perm,:]
    i=0
    count=0
    while(count<val_num):
        v = sum_data[i,0]-1
        id = sum_data[i,1]-1
        if (Istest(test_data,v,id)==0):
            val_data.append(sum_data[i,:])
            count=count+1
        i=i+1
    val_data = np.vstack(val_data)
    print "Successfully create val data! Val data is ",val_data.shape
    return val_data

def ReadTestdata(matfn,testidx):
    test_data=[]
    testsets = sio.loadmat(matfn,variable_names='testsets')
    testsets = testsets['testsets']
    test_data = testsets[testidx,0]
    print "Successfully load test data! Test data is ",test_data.shape
    return test_data

def SaveImgdata(input_data,name):
    if not os.path.exists(outdir+'/'+name):
        os.mkdir(outdir+'/'+name)
    Anum = 0
    Bnum = 0
    for idx in xrange(0,input_data.shape[0]):
        v = input_data[idx,0]-1
        id = input_data[idx,1]-1
        a_num=GetAnum(labeled,v,id)
        b_num=GetBnum(labeled,v,id)
        Anum = Anum + a_num
        Bnum = Bnum + b_num
        for i in xrange(0,a_num):
            a=labeled[v,0][id,i]
            if (a.shape[1]!=0):
                ima = Image.fromarray(np.uint8(a))
                ima = ima.resize((imgw,imgh), Image.ANTIALIAS)
                ima.save(outdir+'/'+name+'/CamA_%04d_%02d.jpg' %(idx+1,i+1))
        for j in xrange(5,5+b_num):
            b=labeled[v,0][id,j]
            if (b.shape[1]!=0):
                imb = Image.fromarray(np.uint8(b))
                imb = imb.resize((imgw,imgh), Image.ANTIALIAS)
                imb.save(outdir+'/'+name+'/CamB_%04d_%02d.jpg' %(idx+1,j+1))

    print 'ids is %d, CamA images is %d, CamB images is %d.' %(input_data.shape[0],Anum,Bnum)

def BuildTripleFile(indir,name,ids):
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
    eachlet = 10;
    
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
    
    pos1_list = open(outdir+'/cuhk03_%s_pos1.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos1_list.write('%s 0\n' %list[x,0])
    pos1_list.close()
    
    pos2_list = open(outdir+'/cuhk03_%s_pos2.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos2_list.write('%s 1\n' %list[x,1])
    pos2_list.close()
    
    neg_list = open(outdir+'/cuhk03_%s_neg.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        neg_list.write('%s 0\n' %list[x,2])
    neg_list.close()
    
    print 'total size of %s data is %d' %(name,list.shape[0])

def BuildRankFile(indir,name,ids):
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
    
    list=[]
    for id in xrange(0,ids):
        for i in xrange(0,int(pairidx[id,0])):
            if not os.path.exists(indir+'/'+name+'/'+'CamA_%04d_%02d.jpg' %(id+1,i+1)): 
                print 'No image is find!'
            count = 0;
            for id_p in xrange(0,ids):
                label=0;
                if id_p==id:
                    label=1;
                #for j in xrange(0,int(pairidx[id_p,1])):
                for j in xrange(0,1):
                    if not os.path.exists(indir+'/'+name+'/'+'CamB_%04d_%02d.jpg' %(id_p+1,j+6)): 
                        print 'No image is find!'
                    list.append(['CamA_%04d_%02d.jpg' %(id+1,i+1),'CamB_%04d_%02d.jpg' %(id_p+1,j+6),label]);
                    count = count + 1;
            #print count;
    
    #perm and write
    list = np.vstack(list)
    #perm = np.random.permutation(list.shape[0])
    #list = list[perm,:]
    
    pos_list = open(outdir+'/cuhk03_%srank_pos.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        pos_list.write('%s %d\n' %(list[x,0],int(list[x,2])))
    pos_list.close()
    
    neg_list = open(outdir+'/cuhk03_%srank_neg.txt' %name, 'w')
    for x in range(0,list.shape[0]):
        neg_list.write('%s %d\n' %(list[x,1],int(list[x,2])))
    neg_list.close()
    
    print 'total size of %s data is %d' %(name,list.shape[0])
       
if __name__ == '__main__':
        
    #four pathes should be set: txtdir,outdir,valdir,indir.

    #output path for the training list and the valuation list
    txtdir = '.../BuildData/cuhk03Data'
    #output path for three folders (train,val,test)
    outdir = '.../BuildData/cuhk03Data/image'
    #the Valdata_list_XX.npy if used our recorded val data
    valdir = '.../BuildData'
    #the cuhk03 source data
    indir='.../cuhk03/cuhk03.mat'
   
    #parameters
    imgw=64
    imgh=128
    start_cam=0
    end_cam=3
    if not os.path.exists(txtdir):
        os.mkdir(txtdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    testidx=0#0~19
    
    #preprocessor
    test_data = ReadTestdata(indir,testidx)
    labeled=sio.loadmat(indir,variable_names='labeled')  
    labeled = labeled['labeled']

    
    #val_data = CreateValdata(labeled,test_data)#Load if produce val data randomly
    val_data = np.load(r'%s/RecordedValdata_cuhk03.npy' %(valdir))#Load if used our recorded val data
    train_data = Preprocessor(labeled,test_data,val_data)

    #save images for each set
    SaveImgdata(train_data,'train')
    SaveImgdata(val_data,'val')
    SaveImgdata(test_data,'test%d' %testidx)

    #save list for training and valuation
    BuildTripleFile(indir,'train',1160)
    BuildRankFile(txtdir,'val',100)#valuation for classfication
    BuildTripleFile(txtdir,'val',100)#valuation for ranking