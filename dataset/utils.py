# -*- coding: utf-8 -*-


import cv2
import numpy as np
import math

def show_landmark(cephaImg,landmarks):
    """
        view cephalometry with landmark for visualization
    """
    for (x,y) in landmarks:
        xx = int(cephaImg.shape[0]*x)
        yy = int(cephaImg.shape[1]*y)
        cv2.circle(cephaImg,(xx,yy),2,(0,0,0),-1)
    cv2.imshow("cephalometry",cephaImg)
    cv2.waitKey(0)
    
def rotate(img,bbox,landmarks,alpha):
    """
        given a cephalometry with bbox and landmark,rotate with alpha
        and return rotated cephalometry with bbox,landmark (absolute position)
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center,alpha,1)
    img_rotated_by_alpha = cv2.warpAffine(img,rot_mat,img.shape)
    landmark_r = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmarks])
    cephaImg_r = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return (cephaImg_r,landmark_r)

def randomShift(landmarkGt,shiftMin,shiftMax):
    """
        Random Shift one time
    """
    '''landmarkP = [0,0]
    radis = np.random.uniform(shiftMin,shiftMax)
    alpha = np.random.random(size=samples_num)*2*np.pi-np.pi 
    i_set = np.arange(0,samples_num,1)
    radis = np.random.uniform(shiftMin,shiftMax)
    for i in i_set:
        len = np.sqrt(np.random.random())*radis
    '''
    
    landmarkP = [0,0]
    radis = np.random.uniform(shiftMin,shiftMax) 
    alpha = np.random.uniform(0,360)
    #print str(radis) + '----------' + str(alpha)
    x = int(radis*math.cos(alpha))
    y = int(radis*math.sin(alpha))
    landmarkP[0] = landmarkGt[0] + x
    landmarkP[1] = landmarkGt[1] + y
    '''
    if math.isnan(landmarkP[0]) or math.isnan(landmarkP[0]):
        print 'dddddddddddddddddddddddddddd'
        randomShift(landmarkGt,shiftMin,shiftMax)'''
    #检验产生的坐标位置是否符合后续取图像块的大小，不合理则继续产生新的位置，直到找到合理的位置
    if landmarkP[0]>20 and landmarkP[0]<625 and landmarkP[1]>20 and landmarkP[1]<780:
        #print '&&&&&&&&&&&&' + str(landmarkP)
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



