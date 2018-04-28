# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:30:36 2017

@author: Administrator
"""

import os
from active_shape_models_he import PointsReader,ActiveShapeModel,ShapeViewer

dirname = 'dataset/data/'

def main():
    
    filename = os.path.join(dirname,'trainListFile.txt')
    
    shapes = PointsReader.readPointsFile(filename)
    print shapes[0].get_normal_to_point(0)
    asm = ActiveShapeModel(shapes)
    ShapeViewer.show_modes_of_variation(asm,int())
    print "Finished!!!"

if __name__ == 'main':
    main()