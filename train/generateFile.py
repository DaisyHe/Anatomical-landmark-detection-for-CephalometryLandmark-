# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 21:13:52 2017

@author: Administrator
"""

from os.path import exists
import os

path = '2__L'

for i in range(1,20):
    pa = path + str(i)
    if not exists(pa):
        os.mkdir(pa)