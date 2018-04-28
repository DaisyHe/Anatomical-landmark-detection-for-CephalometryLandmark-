# -*- coding: utf-8 -*-

"""
    This file use Caffe model to predict landmarks and evaluate the mean error.
"""

import sys
import time
import cv2
import numpy as np
from numpy.linalg import norm
from common import getDataFromTxt,logger
import matplotlib.pyplot as plt

TXT = 'dataset/data/testImageList.txt'
template = '''
    ################ Summary #################
    Test Number: %d
    Time Comsume: %.03f s
    FPS: %.03f
    LEVEL - %d
    Mean Error(the first five): 
        L1 = %f
        L2 = %f
        L3 = %f
        L4 = %f
        L5 = %f
    Failure:
        L1 = %f
        L2 = %f
        L3 = %f
        L4 = %f
        L5 = %f
'''

def evaluateError(landmarkGt,landmarkP,bbox):
    e = np.zeros(19)
    for i in range(19):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    e = e / bbox.w
    print 'landmarkGt:',landmarkGt
    print 'landmarkP:',landmarkP
    print 'error:',e
    return e

def E(level=1):
    if level == 1:
        from common import level1 as P
    elif level == 2:
        from common import level2 as P
        
    data = getDataFromTxt(TXT)
    error = np.zeros((len(data),19))
    for i in range(len(data)):
        imgPath,bbox,landmarkGt = data[i]
        img = cv2.imread(imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        
        landmarkP = P(img,bbox)
        
        #real landmark
        landmarkP = bbox.reprojectLandmark(landmarkP)
        landmarkGt = bbox.reprojectLandmark(landmarkGt)
        error[i] = evaluateError(landmarkGt,landmarkP,bbox)
    return error
    
def plotError(error,name):
    # config global plot
    plt.rc('font',size=16)
    plt.rcParams['savefig.dpi'] = 240
    
    fig = plt.figure(figsize=(20,15))
    binwidth = 0.001
    yCut = np.linspace(0,70,100)
    xCut = np.ones(100)*0.05
    
    for i in range(19):
        ax = fig.add_subplot(321+i)
        data = error[:,i]
        ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
        ax.plot(xCut, yCut, 'r', linewidth=2)
        ax.set_title('L'+(i+1))

    fig.suptitle('%s' % name)
    fig.savefig('log/%s.png'%name)
    


nameMapper = ['level1_test','level2_test']

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    level = int (sys.argv[1])
    
    t = time.clock()
    error = E(level)
    t = time.clock() - t
                  
    N = len(error)
    fps = N / t
    errorMean = error.mean(0)
    
    
    #failure
    failure = np.zeros(19)
    threshold = 0.05
    for i in range(19):
        failure[i] = float(sum(error[:,i] > threshold)) / N
            
    #log string
    s = template % (N, t, fps, level, errorMean[0], errorMean[1], errorMean[2], \
        errorMean[3], errorMean[4], failure[0], failure[1], failure[2], \
        failure[3], failure[4])
    print s
    
    logfile = 'log/{0}.log'.format(nameMapper[level])
    with open(logfile, 'w') as fd:
        fd.write(s)
        
    #plot error hist
    plotError(error,nameMapper[level])
    
