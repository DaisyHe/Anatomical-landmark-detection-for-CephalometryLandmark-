# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:15:57 2017

@author: Administrator
"""
from common import getDataFromTxt,logger,shuffle_in_unison_scary,processImage
from collections import defaultdict
from level2 import getData
from utils import randomShiftWithArgument
import cv2
import numpy as np
import os
from common import drawBBoxLandmark,createDir
import h5py

types =  [(0,'L1'),(1,'L2'),(2,'L3'),(3,'L4'),(4,'L5'),
         (5,'L6'),(6,'L7'),(7,'L8'),(8,'L9'),(9,'L10'),
         (10,'L11'),(11,'L12'),(12,'L13'),(13,'L14'),(14,'L15'),
         (15,'L16'),(16,'L17'),(17,'L18'),(18,'L19'),]

def generateSamples(testData,data,landmarks):
    for (imgPath,bbox) in data:
        img = cv2.imread(imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger('process %s' % imgPath)
        height,width = img.shape[:2]
        #downsampled by 3: 3X3 Patch
        #size = (int(width/3),int(height/3))
        size = (width,height)#不进行下采样
        img = cv2.resize(img,size,interpolation=cv2.INTER_NEAREST)

        testData = getData(testData,landmarks,img)#test此时为一个字典，存储样本点和对应的图像块
        
        print ('After getting raw data,there are %d datas') % len(testData)
        for idx,landmark in enumerate(landmarks):
            # 产生样本点 samples
            landmark_samples = randomShiftWithArgument(landmark,0,100,150)
            testData = getData(testData,landmark_samples,img)
     
    return testData

def geneDataTxt(imgPath,landmarkREF,mode = 'test'):
    
    imgTxt = 'dataset/data/testImageList.txt'
    data = getDataFromTxt(imgTxt,False,False)  #image_path,bbox
    testData = defaultdict(lambda:dict(landmarks=[],patches=[],label=[]))
    logger("generate 25 positive samples and 500 negative samples for per landmark")
    
    testData = generateSamples(testData,data,landmarkREF)
    
    for idx,name in types:
        patches = np.asarray(testData[name]['patches'])
        landmarks = np.asarray(testData[name]['landmarks'])
        #label = np.asarray(testData[name]['label'])

        patches = processImage(patches)
        shuffle_in_unison_scary(patches,landmarks)
        
        createDir('dataset/test/%s' % imgPath[-7:-4])
        
        with h5py.File('dataset/test/%s/%s.h5' % (imgPath[-7:-4],name),'w') as h5:
            h5['data'] = patches.astype(np.float32)
            h5['landmark'] = landmarks.astype(np.float32)
            #h5['label'] = label.astype(np.uint8)
        with open('dataset/test/%s/%s.txt' % (imgPath[-7:-4],name),'w') as fd:
            fd.write('dataset/test/%s/%s.h5'% (imgPath[-7:-4],name))
        
        '''with open('dataset/test/%s.txt' % (name),'w') as fd:
            fd.write(landmarks.astype(np.float32))
            fd.write(patches.astype(np.float32))
            fd.write(label.astype(np.uint8))'''
            
TXT = 'dataset/data/'

if __name__ == '__main__': 
    
    txtFile = os.path.join(TXT,'testImageList.txt')
    data = getDataFromTxt(txtFile,with_landmark=False) #imgPath,bbox
    for imgPath,bbox in data:
        print imgPath
        img = cv2.imread(imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)

        #landmarks = level1(img,bbox)   #正常情况下，应该用level1学习出来的坐标作为初始化
        landmarkREF = [[835,996],[1473,1029],[1289,1279],[604,1228],[1375,1654],
                 [1386,2019],[1333,2200],[1263,2272],[1305,2252],[694,1805],
                 [1460,1870],[1450,1864],[1588,1753],[1569,2013],[1514,1620],
                 [1382,2310],[944,1506],[1436,1569],[664,1340]]#随机选取的坐标,这里用了第一张作为初始化
        landmarkREF = np.array(landmarkREF,dtype=int)
        drawBBoxLandmark(img,bbox,landmarkREF,color=(0,255,0))
        geneDataTxt(imgPath,landmarkREF)#test/L1.txt
           
    
    