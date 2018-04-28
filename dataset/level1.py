# -*- coding: utf-8 -*-

"""
    all data are formated as (data,lanmark),landmark is((x1,y1),(x2,y2)...)  19
"""

import os
from os.path import join,exists
import cv2
import numpy as np
import h5py
from common.utils import shuffle_in_unison_scary,logger,createDir,processImage
from common.utils import getDataFromTxt
from utils import rotate

TRAIN = 'dataset/data/'
OUTPUT = 'train/'

if not exists(OUTPUT): os.mkdir(OUTPUT)
assert(exists(TRAIN) and exists(OUTPUT))

def generate_hdf5(cephaTxt,output,fname,argument=False):
    
    data = getDataFromTxt(cephaTxt)#return [(img_path,landmark,bbox)]   
    cepha_imgs = []
    cepha_landmarks = []
    for (imgPath,landmarkGt,bbox) in data:
        img = cv2.imread(imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        
        #downsampled by 3: 3x3 patch
        height,width = img.shape[:2]
        size = (int(width/3),int(height/3))
        cephaImg = cv2.resize(img,size,interpolation=cv2.INTER_NEAREST)
        
        cepha_bbox = bbox#.subBBox(-0.05,1.05,-0.05,1.05)
        cepha_img = cephaImg[cepha_bbox.top:cepha_bbox.bottom+1,cepha_bbox.left:cepha_bbox.right+1]
    
        if argument and np.random.rand()>-1:
            ###rotation
            if np.random.rand() > 0.5:
                cepha_rotated_alpha,landmark_rotated = rotate(cepha_img,cepha_bbox,bbox.reprojectLandmark(landmarkGt),5)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)#relative
                cepha_rotated_alpha = cv2.resize(cepha_rotated_alpha,(39,39))
                cepha_imgs.append(cepha_rotated_alpha.reshape((1,39,39)))
                cepha_landmarks.append(landmark_rotated.reshape(38))
            if np.random.rand() > 0.5:
                cepha_rotated_alpha,landmark_rotated = rotate(cepha_img,cepha_bbox,bbox.reprojectLandmark(landmarkGt),-5)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                cepha_rotated_alpha = cv2.resize(cepha_rotated_alpha,(39,39))
                cepha_imgs.append(cepha_rotated_alpha.reshape((1,39,39)))
                cepha_landmarks.append(landmark_rotated.reshape(38))
        
        cepha_img = cv2.resize(cepha_img,(39,39))
        cepha_img = cepha_img.reshape((1,39,39))
        cepha_landmark = landmarkGt.reshape((38))
        
        cepha_imgs.append(cepha_img)
        cepha_landmarks.append(cepha_landmark)

    cepha_imgs,cepha_landmarks = np.asarray(cepha_imgs),np.asarray(cepha_landmarks)
    
    cepha_imgs = processImage(cepha_imgs)
    shuffle_in_unison_scary(cepha_imgs,cepha_landmarks)
    
    #save file
    base = join(OUTPUT,'1_cepha')#train/1_cepha   (or test)
    createDir(base)
    output = join(base,fname)#train/1_cepha/train.h5  (or test)
    output = output.replace('\\','/')
    logger("generate %s" % output)
    with h5py.File(output,'w') as h5:
        h5['data'] = cepha_imgs.astype(np.float32)
        h5['landmark'] = cepha_landmarks.astype(np.float32)
    
if __name__ == '__main__':
    #train data
    train_txt = join(TRAIN,'trainImageList.txt')
    generate_hdf5(train_txt,OUTPUT,'train.h5',argument=True)  #数据增广
    
    test_txt = join(TRAIN,'validationImageList.txt')
    generate_hdf5(test_txt,OUTPUT,'test.h5')
    
    with open(join(OUTPUT,'1_cepha/train.txt'),'w') as fd:
        fd.write('train/1_cepha/train.h5')
    with open(join(OUTPUT,'1_cepha/test.txt'),'w') as fd:
        fd.write('train/1_cepha/test.h5')
    