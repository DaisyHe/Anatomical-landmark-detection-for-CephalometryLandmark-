#!/usr/bin/env bash

#generate caffe model
#python prototxt/generate.py

#generate annotation File
#python dataset/data/AnnotationByMD/generateTxtList.py

#generate model file
#python model/generateModelFile.py

#level1
#python dataset/level1.py
#rm -rf log/train1.log
#echo "Train LEVEL_1"
#python train/level.py 1

#level2
python dataset/level2.py
#rm -f log/train2.log
#echo "Train LEVEL_2"
#python train/level.py 2

#shape model
#rm -f log/ASM.log
#echo "Active Shape Model"
#python active_shape_models/fit_to_image_he.py >fit_to_image.log
#python active_shape_models/show_variation_he.py >show

#echo "Finished!!! =.="


#python test/run_draw.py 2 >log/test.log