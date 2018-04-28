# -*- coding: utf-8 -*-

import sys

s1 = ['Cepha']
s2 = ['L1','L2','L3','L4','L5',
      'L6','L7','L8','L9','L10',
      'L11','L12','L13','L14','L15',
      'L16','L17','L18','L19']

def generate(network,level,names,mode='CPU'):
    """
        Generate template
        network: CNN type
        level: LEVEL
        names: CNN names
        mode: CPU or GPU
    """
    assert(mode == 'CPU' or mode == 'GPU')
    #train文件用于训练，solver文件是参数设置，deploy测试的时候使用
    types = ['train','solver','deploy']
    
    for name in names:
        for t in types:
            #template为模板文件
            templateFile = 'prototxt/{0}_{1}.prototxt.template'.format(network,t)
            with open(templateFile,'r') as fd:
                template = fd.read()#读取模板文件内容
                outputFile = 'prototxt/{0}_{1}_{2}.prototxt'.format(level,name,t)
                with open(outputFile,'w') as fd:
                    fd.write(template.format(level=level,name=name,mode=mode))#针对每个不同的点的网络，产生不同的配置文件

if __name__ == '__main__':
    #是GPU训练还是CPU训练
    if len(sys.argv) == 1:
        mode = 'GPU'
    else:
        mode = 'CPU'
        
    #s1训练第一层网络（给第二层进行初始化），这里先不考虑
    #generate('s1',1,s1,mode)
    #s2为训练第二层网络，暂时随机选取一组点第二层网络的初始化
    generate('s2',2,s2,mode)