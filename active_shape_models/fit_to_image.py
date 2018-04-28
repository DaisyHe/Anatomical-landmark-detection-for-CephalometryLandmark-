#!/usr/bin/env python
import sys
import cv
from active_shape_models import PointsReader,ShapeViewer,ActiveShapeModel,ModelFitter

def main():
  shapes = PointsReader.read_directory(sys.argv[1])         #读取一个完整的点的目录，返回一个shapes的列表
  a = ActiveShapeModel(shapes)      #构造匹配一个模型
  # load the image
  i = cv.LoadImage(sys.argv[2])
  m = ModelFitter(a, i)         #将模型拟合到图像中
  ShapeViewer.draw_model_fitter(m)      #在图像上画出模型

  for i in range(1):
    m.do_iteration(3)
    ShapeViewer.draw_model_fitter(m)
  for i in range(1):
    m.do_iteration(2)
    ShapeViewer.draw_model_fitter(m)
  for i in range(10):
    m.do_iteration(3)
    ShapeViewer.draw_model_fitter(m)
  for j in range(100):
    m.do_iteration(0)
    ShapeViewer.draw_model_fitter(m)

if __name__ == "__main__":
  main()

