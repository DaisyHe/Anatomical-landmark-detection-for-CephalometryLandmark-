# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
from os.path import join,exists

def logger(msg): 
    """
        log message
    """
    now = time.ctime()
    print("[%s] %s" % (now,msg))
    
def createDir(path):
    if not exists(path):
        os.mkdir(path)
          
def shuffle_in_unison_scary(a,b,c=None):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    if c is not None:
        np.random.set_state(rng_state)
        np.random.shuffle(c)
          
def drawBBoxLandmark(img,bbox,landmarks,color):
    #draw bbox
    cv2.rectangle(img,(bbox.left,bbox.top),(bbox.right,bbox.bottom),(255,0,0),2)
    #draw landmarks
    for x,y in landmarks:
        cv2.circle(img,(int(x),int(y)),2,color,-1)
    return img
          
def drawLandmark(img,landmark,N=81):
    #draw landmark
    cv2.circle(img,(int(landmark.x),(int(landmark.y))),2,(0,255,0),-1)
    #left = int(landmark.x-N/2)
    #right = int(landmark.x+N/2)
    #bottom = int(landmark.y+N/2)
    #top = int(landmark.y-N/2)
    #draw box
    #cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
    return img
          
def getDataFromTxt(txt,with_landmark=True,raw=False):
    """
    Generate data from txt file
    return [(img_path,landmark,bbox)]
    landmark:[(x1,y1),(x2,y2),...]
    """
    dirname = os.path.dirname(txt)
    with open(txt,'r') as fd:
        lines = fd.readlines()
          
    result = []
    for line in lines:
        line = line.strip()
        #print line
        components = line.split(' ')
        #print components
        img_path = join(dirname,components[0]) #file path
        img_path = img_path.replace('\\','/')
        #bounding box
        #bbox = (0,645,0,800)
        bbox = (0,2400,0,1935)
        bbox = [int(_) for _ in bbox]
        # landmark
        if not with_landmark:
            result.append((img_path,BBox(bbox)))
            continue
        landmark = np.zeros((19,2))
        for i in range(0,19):
            #下采样
            #x = int(float(components[1+2*i])/3)#这样写的原因是不能有str转化为int，但是能由str转化为float，很奇怪
            #y = int(float(components[1+2*i+1])/3)
            # 原始图像
            x = components[1+2*i]
            y = components[1+2*i+1]
            #print x,y
            rv = (y,x)
            landmark[i] = rv
        if raw == False:
            for i,one in enumerate(landmark):
                rv = ((one[0]-bbox[0])/(bbox[1]-bbox[0]),(one[1]-bbox[2])/(bbox[3]-bbox[2]))  #归一化
                landmark[i] = rv #相对坐标
        #print (img_path,landmark,BBox(bbox))
        result.append((img_path,landmark,BBox(bbox)))
    return result
    
def getLandmarks(txt):
    """
        Generate data from txt file
        return [(img_path,landmark)]
    """
    with open(txt,'r') as fd:
        lines = fd.readlines()
        
    landmarks = []
    i = 0
    for line in lines:
        if i == 19:
            i += 1
            break
        line = line.strip()
        components = line.split(',')
        landmarks.append(components[0]).append(components[1])
    return landmarks
    
          
def getLandmarkPatchAndBBox(img,landmark,N):
    """
    Get a patch image around landmark in bbox
    landmark:relative_point in [0,1] in box
    """
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
    patch_bbox = BBox([left,right,top,bottom])
    return patch,patch_bbox
          
def processImage(imgs):
    """
    process images before feeding to CNNs
    imgs:L*1*W*H
    """
    imgs = imgs.astype(np.float32)
    for i,img in enumerate(imgs):
        m = img.mean()
        s = img.std()
        imgs[i] = (img - m)/(s+1)  #分母为s时出现RuntimeWarning: invalid value encountered in divide
    return imgs
          
def dataArgument(data):
    """
    data Arguments
    data:
    imgs:L*1*W*H
    landmarks:L * 19*2
    """
    pass
          
class BBox(object):
    """Bounding Box of cephalometry"""
    def __init__(self,bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1]-bbox[0]
        self.h = bbox[3]-bbox[2]
          
    def expand(self,scale=0.05):
        bbox = [self.left,self.right,self.top,self.bottom]
        bbox[0] -= int(self.w*scale)
        bbox[1] += int(self.w*scale)
        bbox[2] -= int(self.h*scale)
        bbox[3] += int(self.h*scale)
        return BBox(bbox)
          
    def project(self,point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x,y])
          
    def reproject(self,point):
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x,y])
          
    def reprojectLandmark(self,landmarks):
        p = np.zeros((len(landmarks),2))
        for i in range(len(landmarks)):
            p[i] = self.reproject(landmarks[i])
        return p
          
    def projectLandmark(self,landmarks):
        p = np.zeros((len(landmarks),2))
        for i in range(len(landmarks)):
            p[i] = self.project(landmarks[i])
        return p
          
    def subBBox(self,leftR,rightR,topR,bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR 
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])