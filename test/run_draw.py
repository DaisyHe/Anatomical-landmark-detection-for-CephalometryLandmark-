# -*- coding: utf-8 -*-

"""
    This file use Caffe model to prefict data
"""

import os,sys 
import cv2
#from common import level1,level2
from testUtils import getDataFromTxt#,drawLandmark,drawBBoxLandmark
 


#def getTestData(h5file):
 

if __name__ == '__main__':
    
    level = 2 #int(sys.argv[1])
    
    OUTPUT = 'test/out_{0}'.format(level)
    #createDir(OUTPUT)
    txtFile = os.path.join('dataset/data/','testImageList.txt')
    
    #这时可以不用读取landmark
    data = getDataFromTxt(txtFile,with_landmark=False) #imgPath
    for imgPath,bbox in data:
        #读取灰度图
        img = cv2.imread(imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)

        if level == 2:
            #landmarks = level1(img,bbox)   #正常情况下用学习出来的坐标进行初始化，测代码时先用一组随机选取的数据初始化
            landmarkREF = [[835,996],[1473,1029],[1289,1279],[604,1228],[1375,1654],
                 [1386,2019],[1333,2200],[1263,2272],[1305,2252],[694,1805],
                 [1460,1870],[1450,1864],[1588,1753],[1569,2013],[1514,1620],
                 [1382,2310],[944,1506],[1436,1569],[664,1340]]#随机选取的坐标,这里用了第一张作为初始化
            print landmarkREF[0]
            #cv2.circle(img,(int(landmark[0].x),(int(landmark[0].y))),2,(0,255,0),-1)
            
            