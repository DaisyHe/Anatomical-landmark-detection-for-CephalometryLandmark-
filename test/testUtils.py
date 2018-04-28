# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:14:23 2018

@author: Administrator
"""

import os
from os.path import join
import numpy as np

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
        #print (img_path,landmark,BBox(bbox))
        result.append((img_path,landmark))
    return result