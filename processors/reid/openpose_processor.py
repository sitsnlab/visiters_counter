# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:30:59 2024

@author: ab19109
オープンキャンパス用のOpenPose
"""
import sys
import os
import os.path as osp

dir_path = os.path.dirname(os.path.realpath(__file__))
#print("dir path > ", dir_path)
#dir_path = C:\Users\ab19109\opencampus\visitor-counter\processors\reid
sys.path.append(dir_path + r'\openpose\build\python\openpose\Release');
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + r'\openpose\build\x64\Release;' + \
    dir_path + r'\openpose\build\bin;'

#print("os environ > ", os.environ['PATH'])
#print("sys path > ", sys.path)
import pyopenpose as op

