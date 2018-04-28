# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 20:04:33 2018

@author: Administrator
"""

from common import getDataFromTxt,logger,shuffle_in_unison_scary,processImage
from collections import defaultdict
from level2 import getTrainData
from utils import randomShiftWithArgument
import cv2
import numpy as np

types =  [(0,'L1'),(1,'L2'),(2,'L3'),(3,'L4'),(4,'L5'),
         (5,'L6'),(6,'L7'),(7,'L8'),(8,'L9'),(9,'L10'),
         (10,'L11'),(11,'L12'),(12,'L13'),(13,'L14'),(14,'L15'),
         (15,'L16'),(16,'L17'),(17,'L18'),(18,'L19'),]

def geneDataTxt(imgPath,landmarks):
    
    imgTxt = 'dataset/data/testImageList.txt'
    data = getDataFromTxt(imgTxt,False,False)  #image_path,bbox
    trainData = defaultdict(lambda:dict(landmarks=[],patches=[],label=[]))
    logger("generate 25 positive samples and 500 negative samples for per landmark")
    
    trainData = generateSamples(trainData,data,landmarks)
    
    arr = []
    for idx,name in types:
        patches = np.asarray(trainData[name]['patches'])
        landmarks = np.asarray(trainData[name]['landmarks'])
        label = np.asarray(trainData[name]['label'])
        
        arr1 = landmarks.reshape(1,-1)
        arr.append(arr1)
        
        patches = processImage(patches)
        shuffle_in_unison_scary(patches,landmarks,label)
        
        with open('test/%s.txt' % (name),'w') as fd:
            fd.append(landmarks.astype(np.float32))
            fd.append(patches.astype(np.float32))
            fd.append(label.astype(np.uint8))
    
    
def generateSamples(trainData,data,landmarks):
    t = 0
    for (imgPath,bbox) in data:
        img = cv2.imread(imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger('process %s' % imgPath)
        height,width = img.shape[:2]
        #downsampled by 3: 3X3 Patch
        size = (int(width/3),int(height/3))
        img = cv2.resize(img,size,interpolation=cv2.INTER_NEAREST)
        
        trainData,t = getTrainData(trainData,landmarks,img)
        
        print ('After getting raw data,there are %d datas') % t
              
        r2 = 20 / 3
        r3 = 10 #400 / 3
        
        for idx,landmark in enumerate(landmarks):
            print '@@@@@@@@@@@@@@' + str(idx)
            # 25 Positive samples
            landmarkPs25 = randomShiftWithArgument(landmark,0,r2,25)
            trainData,t = getTrainData(trainData,landmarkPs25,img)
            
            #print ('After getting 25 positive samples,there are %d datas') % t
            # 500 negative samples
            landmarkNs500 = randomShiftWithArgument(landmark,r2,r3,500)
            trainData,t = getTrainData(trainData,landmarkNs500,img)
            
            print ('After getting 25 positive and 500 negative samples,there are %d datas') % t
    return trainData
    
    


if __name__ == '__main__':
    landmarkREF = [[835,996],[1473,1029],[1289,1279],[604,1228],[1375,1654],
                 [1386,2019],[1333,2200],[1263,2272],[1305,2252],[694,1805],
                 [1460,1870],[1450,1864],[1588,1753],[1569,2013],[1514,1620],
                 [1382,2310],[944,1506],[1436,1569],[664,1340]]#随机选取的坐标,这里用了第一张作为初始化
    #drawBBoxLandmark(img,bbox,landmarkREF,color=(0,255,0))
    geneDataTxt(imgPath,landmarkREF)#test/L1.txt
            
    landmarks = []
    probs = []
    labels = []
                       
    txtPath = 'test/'
    for i in range(1,20):
        txt = os.path.join(txtPath,'L%d.txt' % i)
        data = getTestData(txt)#读取文件数据，该数据包含采样的坐标位置、图像块、标签
                    
        for (landmarkRef,imgPatch,label) in data:
            landmark,prob,label = P(landmarkRef,imgPatch,label) #level2——landmark,prob,label
                
            landmarks.append(landmark)
            probs.append(prob)
            labels.append(label)
            
            result = pd.DataFrame({'landmark':landmarks,'prob':probs,'label':labels})
            result.to_csv('result_L%d.csv' % i)