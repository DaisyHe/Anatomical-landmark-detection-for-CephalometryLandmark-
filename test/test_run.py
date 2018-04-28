# -*- coding: utf-8 -*-

"""
    This file use Caffe model to prefict data
"""

import os
from os.path import join
import cv2
import numpy as np
import caffe
import math
#from common import level1,level2
#from testUtils import getDataFromTxt#,drawLandmark,drawBBoxLandmark
#from common.cnns import getCNNs
 
def getDataFromTxt(txt,with_landmark=True):
    dirname = os.path.dirname(txt)
    with open(txt,'r') as fd:
        lines = fd.readlines()
          
    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = join(dirname,components[0]) #file path
        img_path = img_path.replace('\\','/')
        bbox = (0,2400,0,1935)
        bbox = [int(_) for _ in bbox]
        # landmark
        if not with_landmark:
            result.append((img_path))
            continue
        landmark = np.zeros((19,2))
        for i in range(0,19):
            x = components[1+2*i]
            y = components[1+2*i+1]
            rv = (y,x)
            landmark[i] = rv
        result.append((img_path,landmark))
    return result

def getCNNs(level=1):
    
    types = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10',
             'L11','L12','L13','L14','L15','L16','L17','L18','L19']
    cnns = []
    for i,t in enumerate(types):
        cnns[i].append(caffe.Net('prototxt/2_%s_deploy.prototxt'%t,'model/2_%s/_iter_100000.caffeModel'%(t)),caffe.TEST)
    return cnns

def randomShift(landmarkGt,shiftMin,shiftMax):
    
    landmarkP = [0,0]
    radis = np.random.uniform(shiftMin,shiftMax) 
    alpha = np.random.uniform(0,360)
    #print str(radis) + '----------' + str(alpha)
    x = int(radis*math.cos(alpha))
    y = int(radis*math.sin(alpha))
    landmarkP[0] = landmarkGt[0] + x
    landmarkP[1] = landmarkGt[1] + y
             
    if landmarkP[1]>20 and landmarkP[1]<1915 and landmarkP[0]>20 and landmarkP[0]<2380:
        return landmarkP
    else:
        randomShift(landmarkGt,shiftMin,shiftMax)
        
def randomShiftWithArgument(landmarkGt,shiftMin,shiftMax,samples_num):
    """
        Random Shift more
    """
    landmarkPs = np.zeros((samples_num,2))
    for i in range(samples_num):
        landmarkPs[i] = randomShift(landmarkGt,shiftMin,shiftMax)
        #print landmarkPs[i]
    return landmarkPs

def getTestPatch(img,landmark,N=81):
    n = N/2
    if landmark[1]<=n or landmark[1]>=(1935-n) or landmark[0]<=n or landmark[0]>=(2400-n):
        radis = N/4
    else:
        radis = N/2
    #landmark[0]为行数（top,bottom）,landmark[1]为列数（left,right）
    left = (landmark[1]-radis).astype('int16')
    right = (landmark[1]+radis).astype('int16')
    top = (landmark[0]-radis).astype('int16') 
    bottom = (landmark[0]+radis).astype('int16')
    patch = img[top:bottom+1,left:right+1]
    return patch   
    
def forward2(self,data,layer='fc2'):
        fake = np.zeros((len(data),1,1,1))
        self.cnn.set_input_arrays(data.astype(np.float32),fake.astype(np.float32))
        self.cnn.forward()
        prob = self.cnn.blobs[layer].data[0]

        t = lambda x: np.asarray([np.asarray([x[2*i],x[2*i+1]]) for i in range(len(x)/2)])
        result = t(data)
        return result,prob
    
def runTest(img,landmarkGt,landmarkREF,N=81):
    cnns = getCNNs(2)#获得19个点的网络结构以及caffemodel
    for idx,center in enumerate(landmarkREF):#19个参考点依次运行
        candidates = []
        landmark_samples = randomShiftWithArgument(center,r1=0,r3=400,samples_numbers=525)
        for center in landmark_samples:
            patch = getTestPatch(img,center,N)
            patch = cv2.resize(patch,(N,N))
            patch = patch.reshape((1,N,N))
            
            #网络层的输入是blobs的
            img.reshape((1,1,N,N))
            result,prob = forward2(img,cnns)
            #满足条件的就加入候选点
            if prob[0] < prob[1]:
                print prob
                candidates.append(center)
            
        cv2.circle(img,(center[1],center[0]),3,(0,0,255),-1)
        
        #将每个点对应的候选点保存至对应文件
        candidates = np.asarray(candidates)
        with open('test/out_2/L_%d.txt'%(idx+1),'a') as fd:
            fd.write(candidates)

if __name__ == '__main__':
    
    level = 2 #int(sys.argv[1])
    
    OUTPUT = 'test/out_{0}'.format(level)
    #createDir(OUTPUT)
    txtFile = os.path.join('dataset/data/','testImageList.txt')
    
    #这时可以不用读取landmark
    data = getDataFromTxt(txtFile,with_landmark=True) #imgPath
    for imgPath,landmarkGt in data:
        #读取灰度图
        img = cv2.imread(imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)

        if level == 2:
            #landmarks = level1(img,bbox)   #正常情况下用学习出来的坐标进行初始化，测代码时先用一组随机选取的数据初始化
            landmarkREF = [[835,996],[1473,1029],[1289,1279],[604,1228],[1375,1654],
                 [1386,2019],[1333,2200],[1263,2272],[1305,2252],[694,1805],
                 [1460,1870],[1450,1864],[1588,1753],[1569,2013],[1514,1620],
                 [1382,2310],[944,1506],[1436,1569],[664,1340]]#随机选取的坐标,这里用了第一张作为初始化
            #可视化第一个点
            #cv2.circle(img,(int(landmarkREF[0][0]),(int(landmarkREF[0][1]))),3,(0,255,0),-1)#话初始坐标点
            #cv2.circle(img,(int(landmarkGt[0][1]),int(landmarkGt[0][0])),3,(255,0,0),-1)#画正确的坐标点
            #cv2.imwrite('test_1.bmp',img)
            runTest(img,landmarkGt,landmarkREF)
            
            
            