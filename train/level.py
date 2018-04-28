# -*- coding: utf-8 -*-
"""
    This file train Caffe CNN models
"""

import os,sys
import multiprocessing

pool_on = False

models = [
        ['Cepha'],
        ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10',
        'L11','L12','L13','L14','L15','L16','L17','L18','L19'],
        ]

def w(c):
    if c != 0:
        print'\n'
        print':-('
        print'\n'
        sys.exit()

def runCommand(cmd):
    w(os.system(cmd))

def train(level=1):
    """
        train caffe model
    """
    cmds = []
    for model in models[level-1]: #level为2时，读取的是19个点的信息
        #创建存放各个点的model的文件夹
        filename = 'model/{0}_{1}'.format(level,model)
        if not os.path.exists(filename):
            os.mkdir(filename)   
        #caffe的运行语句，下面执行的时候用
        cmd = 'caffe train --solver prototxt/{0}_{1}_solver.prototxt'.format(level,model) 
        cmds.append(cmd)
    if level==2 and pool_on:#并行执行第二层
        pool_size = 3
        pool = multiprocessing.Pool(processes = pool_size,maxtasksperchild=2)
        pool.map(runCommand,cmds)#对每一条命令，都执行
        pool.close()
        pool.join()
    else:
        for cmd in cmds:
            runCommand(cmd)

if __name__ == '__main__':
    argc = len(sys.argv)    #nohup python train/level .py 1 pool_on  ##sys.argv=['train/level.py','1']
    assert(2 <= argc <= 3)
    if argc == 2:
        pool_on = False  #multiproccessing
    #目前只训练第二层的数据，也就是只学习标志点概率那一块
    level = int(sys.argv[1])
    if 1 <= level <= 2:  # 1 or 2
        train(level)
    else:
        for level in range(1,3): # 1 and 2
            train(level)