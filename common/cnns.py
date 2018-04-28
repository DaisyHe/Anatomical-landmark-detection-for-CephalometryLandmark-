# -*- coding: utf-8 -*-

import caffe
import numpy as np

class CNN(object):
    """
        Generalized CNN for simple run forward with given Model
    """
    def __init__(self,net,model):
        self.net = net
        self.model = model
        try:
            self.cnn = caffe.Net(str(net),str(model),caffe.TEST)  #failed if not exist
        except:
            #silence
            print "Can not open %s,%s"%(net,model)
        
    def forward1(self,data,layer='fc2'):
        fake = np.zeros((len(data),1,1,1))
        self.cnn.set_input_arrays(data.astype(np.float32),fake.astype(np.float32))
        self.cnn.forward()
        result = self.cnn.blobs[layer].data[0]
        t = lambda x: np.asarray([np.asarray([x[2*i],x[2*i+1]]) for i in range(len(x)/2)])
        result = t(result)
        return result
    
    def forward2(self,data,layer='fc2'):
        fake = np.zeros((len(data),1,1,1))
        self.cnn.set_input_arrays(data.astype(np.float32),fake.astype(np.float32))
        self.cnn.forward()
        prob = self.cnn.blobs[layer].data[0]

        t = lambda x: np.asarray([np.asarray([x[2*i],x[2*i+1]]) for i in range(len(x)/2)])
        result = t(data)
        return result,prob
        
# global cnns
cnn = dict(level1=None,level2=None)
m1 = '_iter_1000000.caffeModel'
m2 = '_iter_100000.caffeModel'

def getCNNs(level=1):
    types = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','L11','L12','L13','L14','L15','L16','L17','L18','L19']
    if level == 1:
        if cnn['level1'] is None:
            cnn['level1']=[]
            cnn['level1'].append(CNN('prototxt/1_cepha_deploy.prototxt','model/1_cepha/%s'%(m1)))
        return cnn['level1']
    elif level == 2:
        if cnn['level2'] is None:
            cnn['level2']=[]
            for t in types:
                cnn['level2'].append(CNN('prototxt/2_%s_deploy.prototxt'%t,'model/2_%s/%s'%(t,m2)))
        return cnn['level2']








