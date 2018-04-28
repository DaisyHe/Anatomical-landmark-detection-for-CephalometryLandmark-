# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:45:04 2017

@author: Administrator
"""

from active_shape_models_he import PointsReader,ActiveShapeModel,ModelFitter,ShapeViewer
import cv2
import os
from common import logger

dirname = 'dataset/data/' 

def main():
    
    pointfiledir = os.path.join(dirname,'AnnotationByMD/400_senior/')
    imgdir = os.path.join(dirname,'TrainData')
    
    logger("generate active shape model!!")
    
    shapes = PointsReader.read_directory(pointfiledir)
    asm = ActiveShapeModel(shapes) 
    
    logger("draw model fitter on the given images!")
    for i in range(1,2):
        s = str(i).zfill(3) + '.bmp'
    
        # load the image
        img = cv2.imread(os.path.join(imgdir,s),cv2.CV_LOAD_IMAGE_GRAYSCALE)
        model = ModelFitter(asm,img)
        ShapeViewer.draw_model_fitter(model)
        
        for j in range(100):
            model.do_iteration(0)
            ShapeViewer.draw_model_fitter(model)

if __name__ == '__main__':
    main()
