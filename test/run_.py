# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 20:03:14 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-

"""
    This file use Caffe model to prefict data
"""

import os,sys 
import cv2
from common import level1,level2
from common import createDir,getDataFromTxt,logger,drawLandmark,drawBBoxLandmark
import pandas as pd

TXT = 'dataset/data/'

def getTestData(txt):
    f = open(txt)
    landmark,patch,label = f.readline()
    return landmark,patch,label
    #for landmark,patch,label in open(txt):
        #return landmark,patch,label

if __name__ == '__main__':
    
    level = int(sys.argv[1])
    if level == 1:
        P = level1
    elif level == 2:
        P = level2    
    
    OUTPUT = 'dataset/test/out_{0}'.format(level)
    createDir(OUTPUT)
    txtFile = os.path.join(TXT,'testImageList.txt')
    data = getDataFromTxt(txtFile,with_landmark=False) #imgPath,bbox
    for imgPath,bbox in data:
        print imgPath
        img = cv2.imread(imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        
        if level == 1:
            landmark = P(img,bbox) #level1——landmark 
            landmark = bbox.reprojectLandmark(landmark)
            drawBBoxLandmark(img,bbox,landmark)
            drawLandmark(img,bbox,landmark)
            cv2.imwrite(os.path.join(OUTPUT,os.path.basename(imgPath)),img)
            
            
        if level == 2:
            #landmarks = level1(img,bbox)   #学习出的坐标
            landmarkREF = [[835,996],[1473,1029],[1289,1279],[604,1228],[1375,1654],
                 [1386,2019],[1333,2200],[1263,2272],[1305,2252],[694,1805],
                 [1460,1870],[1450,1864],[1588,1753],[1569,2013],[1514,1620],
                 [1382,2310],[944,1506],[1436,1569],[664,1340]]#随机选取的坐标,这里用了第一张作为初始化
            drawBBoxLandmark(img,bbox,landmarkREF,color=(0,255,0))
            dataset.geneDataTxt(imgPath,landmarkREF)#test/L1.txt
            
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
            
            
                    #readData = pd.read_csv('result.csv')
            
                    landmark = bbox.reprojectLandmark(landmark)
                    drawBBoxLandmark(img,bbox,landmark,color=(0,0,255))
                    cv2.imwrite(os.path.join(OUTPUT,os.path.basename(imgPath)),img)
            
            