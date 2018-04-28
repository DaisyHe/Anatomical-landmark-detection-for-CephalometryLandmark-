# -*- coding: utf-8 -*-

import cv2
import numpy as np
from .cnns import getCNNs
from .utils import processImage

def level1(img,bbox):
    """
        img: gray image
        bbox: bounding box
    """
    cephaCNN = getCNNs(level=1)
    
    cepha_bbox = bbox.subBBox(-0.05,1.05,-0.05,1.05)
    cepha_img = img[cepha_bbox.top:cepha_bbox.bottom+1,cepha_bbox.left:cepha_bbox.rigth+1]
    cepha_img = cv2.resize(cepha_img,(300,300))
    
    cepha_img = cepha_img.reshape((1,1,300,300))
    cepha_img = processImage(cepha_img)
    cepha_landmark = cephaCNN.forward1(cepha_img)
    
    landmark = np.zeros((19,2))
    for i in range(0,19):
        landmark[i] = cepha_landmark[i]
    return landmark

def level2(data):
    """
        LEVEL-2
        img: I(xi,yi)
        landmark: m
    """
    cnns = getCNNs(2)
    landmark,prob = cnns.forward2(data)
    if prob>0.5:
        label = 1
    else:
        label = 0
    return landmark,prob,label

    
    
    
    
    
    
    
    
    
    
    
