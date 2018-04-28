# -*- coding: utf-8 -*-

"""
    This file convert data for LEVEL_2 training data.
    all data are formated as (image patch around the landmark),landmark:L1~L19
"""

import time
import numpy as np
from common import getDataFromTxt,getLandmarkPatchAndBBox,processImage
from common import shuffle_in_unison_scary
from collections import defaultdict
import cv2
from common import logger,createDir
import h5py
from utils import randomShiftWithArgument

types =  [(0,'L1'),(1,'L2'),(2,'L3'),(3,'L4'),(4,'L5'),
         (5,'L6'),(6,'L7'),(7,'L8'),(8,'L9'),(9,'L10'),
         (10,'L11'),(11,'L12'),(12,'L13'),(13,'L14'),(14,'L15'),
         (15,'L16'),(16,'L17'),(17,'L18'),(18,'L19'),]

for t in types:
    d = 'train/2_%s' % t[1]
    createDir(d)

def getData(data,landmarks_samples,img,t,N=81):
    for idx,landmark in enumerate(landmarks_samples):
        #print landmark
        patch,patch_bbox = getLandmarkPatchAndBBox(img,landmark,N)
        t += 1
        print '----------------------------------------------------------------------------'
        print patch.shape
        #patch = cv2.resize(patch,(N,N))
        patch = patch.reshape((1,N,N))
        label = 1
        #ldmk = patch_bbox.project(bbox.reproject(landmarkGt[idx]))
        name = "L"+str(idx)
        data[name]['patches'].append(patch)
        data[name]['landmarks'].append(landmark)
        data[name]['label'].append(label)
    return data,t

def generateSamples(trainData,data):
    t = 0   
    print '##########################################################################'
    for (imgPath,landmarkGt,bbox) in data:
        img = cv2.imread(imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        height,width = img.shape[:2]
        #downsampled by 3: 3x3 patch
        cephaImg = cv2.resize(img,(int(width/3),int(height/3)),interpolation=cv2.INTER_NEAREST)
        
        #raw data 
        #trainData,t = getData(trainData,landmarkGt,cephaImg,t)
        #print ('After getting raw data,there are %d datas') % t
        
        r1 = 20 / 3
        r2 = 20 #60/3
        r3 = 400 #400/3
        for idx,landmark in enumerate(landmarkGt):#19个landmark
            # 25 Positive samples
            landmarkPs25 = randomShiftWithArgument(landmark,0,r1,25)
            trainData,t = getData(trainData,landmarkPs25,cephaImg,t)
            print ('After getting 25 positive samples,there are %d datas') % t
            # 500 negative samples
            landmarkNs500 = randomShiftWithArgument(landmark,r2,r3,500)
            trainData,t = getData(trainData,landmarkNs500,cephaImg,t)
            print ('After getting 25 positive and 500 negative samples,there are %d datas') % t
            
            if idx == 1:
                break
    return trainData

def generate(txt,mode,N):
    """
        generate Training Data for LEVEL-2: patches around landmarks
        mode: train or validate
    """
    assert(mode == 'train')
    #从txt文件中获取数据
    data = getDataFromTxt(txt,True,True) #return [(image_path,landmarks,bbox)]
    #用defaultdict存储数据
    trainData = defaultdict(lambda:dict(landmarks=[],patches=[],label=[]))
    logger("generate 25 positive samples and 500 negative samples for per landmark")
    
    #产生正负训练样本
    trainData = generateSamples(trainData,data)
    print 'All data:' + str(len(trainData))
    #print trainData['L1']
    #arr = []

    for idx,name in types:#19个点
        logger('writing training data of %s' % name)
        patches = np.asarray(trainData[name]['patches'])
        landmarks = np.asarray(trainData[name]['landmarks'])
        label = np.asarray(trainData[name]['label'])
        
        #arr1 = landmarks.reshape(1,-1)
        #arr.append(arr1)
        
        patches = processImage(patches)       
        shuffle_in_unison_scary(patches,landmarks,label)
        
        #将训练数据保存为hdf5格式
        with h5py.File('train/2_%s/%s.h5' % (name,mode),'w') as h5:
            h5['data'] = patches.astype(np.float32)
            h5['landmark'] = landmarks.astype(np.float32)
            h5['label'] = label.astype(np.uint8)
        with open('train/2_%s/%s.txt' % (name,mode),'w') as fd:
            fd.write('train/2_%s/%s.h5'% (name,mode))
        #暂时只产生一个点的数据
        if idx == 1:
            break
    #data = pd.DataFrame(arr)
    #data.to_csv('landmark.csv')

if __name__ == '__main__':
    np.random.seed(int(time.time()))
    #N为图像块大小
    N = 81
    #产生训练集
    generate('dataset/data/trainImageList.txt','train',N)  #landmark,box
    
            