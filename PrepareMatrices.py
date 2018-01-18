# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:57:08 2018

@author: zhiyong
"""

import torch.utils.data as utils
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import time
import matplotlib.pyplot as plt

# from TrainTestDataset import *

def GetAdjacencyMatrix(sensors):
    fea_size = sensors.shape[0]
    A = np.zeros((fea_size,fea_size))
    A = A + np.eye(fea_size)
    
    # connect adjacent sensors on same roads
    for idx in range(fea_size-1):
        i = idx
        j = idx + 1
        loop_i = sensors[i]
        loop_j = sensors[j]
        if loop_i[:4] == loop_j[:4]:
            A[i,j] = 1
            A[j,i] = 1
            
    # connect adjacent sensors in the intesection areas
    # I-5 & SR-520
    A[138,199] = 1
    A[138,36] = 1
    A[302,137] = 1
    A[302,198] = 1
    # I-405 & SR-520
    A[282,317] = 1
    A[282,150] = 1
    A[152,283] = 1
    A[152,120] = 1
    A[121,317] = 1
    A[121,150] = 1
    A[315,120] = 1
    A[315,283] = 1
    # I-5 & I-90
    A[189,234] = 1
    A[73,190] = 1
    A[73,25] = 1
    A[27,234] = 1
    # I-405 & I-90
    A[274,250] = 1
    A[274,87] = 1
    A[90,276] = 1
    A[90,112] = 1
    A[114,250] = 1
    A[114,87] = 1
    A[274,112] = 1
    A[274,276] = 1
    
    for idx in range(fea_size-1):
        i = idx
        j = idx + 1
        if A[i,j] == 1:
            A[j,i] = 1
        if A[j,i] == 1:
            A[i,j] = 1
    return A

def GetDirectedAdjacencyMatrix(sensors):
    fea_size = sensors.shape[0]
    A = np.zeros((fea_size,fea_size))
    A = A + np.eye(fea_size)
    
    # connect adjacent sensors on same roads
    for idx in range(fea_size-1):
        i = idx
        j = idx + 1
        loop_i = sensors[i]
        loop_j = sensors[j]
        if loop_i[:4] == loop_j[:4]:
            A[i,j] = 1
            A[j,i] = 1
            
    # connect adjacent sensors in the intesection areas
    # I-5 & SR-520
    A[138,199] = 1
    A[138,36] = 1
    A[302,137] = 1
    A[302,198] = 1
    # I-405 & SR-520
    A[282,317] = 1
    A[282,150] = 1
    A[152,283] = 1
    A[152,120] = 1
    A[121,317] = 1
    A[121,150] = 1
    A[315,120] = 1
    A[315,283] = 1
    # I-5 & I-90
    A[189,234] = 1
    A[73,190] = 1
    A[73,25] = 1
    A[27,234] = 1
    # I-405 & I-90
    A[274,250] = 1
    A[274,87] = 1
    A[90,276] = 1
    A[90,112] = 1
    A[114,250] = 1
    A[114,87] = 1
    A[274,112] = 1
    A[274,276] = 1
    
    for idx in range(fea_size-1):
        i = idx
        j = idx + 1
        if A[i,j] == 1:
            A[j,i] = 1
        if A[j,i] == 1:
            A[i,j] = 1
    return A

def GetDirectedAdjacencyMatrix(sensors):
    fea_size = sensors.shape[0]
    A = np.zeros((fea_size,fea_size))
    A = A + np.eye(fea_size)
    
    D = np.ones((fea_size,fea_size)) * 100
    np.fill_diagonal(D, 0.0001) 
    
    # connect adjacent sensors on same roads
    for idx in range(fea_size-1):
        i = idx
        j = idx + 1
        loop_i = sensors[i]
        loop_j = sensors[j]
        if loop_i[:4] == loop_j[:4]:
            if loop_i[:1] == 'i':
                A[i,j] = 1
                D[i,j] = (float(loop_j[6:11]) - float(loop_i[6:11])) / 100
            else:
                A[j,i] = 1
                D[j,i] = (float(loop_j[6:11]) - float(loop_i[6:11])) / 100
                
            
    # connect adjacent sensors in the intesection areas
    # I-5 & SR-520
    A[138,199], A[138,36], A[37,302], A[198,302] = 1, 1, 1, 1
    D[138,199], D[138,36], D[37,302], D[198,302] = 0.4, 0.4, 0.3, 0.3
    # I-405 & SR-520
    A[282,317], A[282,150] = 1, 1
    D[282,317], D[282,150] = 0.5, 1.1
    A[152,283], A[152,120] = 1, 1
    D[152,283], D[152,120] = 1.7, 0.8
    A[121,317], A[121,150] = 1, 1
    D[121,317], D[121,150] = 1.7, 1.7
    A[315,120], A[315,283] = 1, 1
    D[315,120], D[315,283] = 0.8, 1.8
    # I-5 & I-90
    A[189,234], A[73,190], A[73,25], A[27,234] = 1, 1, 1, 1
    D[189,234], D[73,190], D[73,25], D[27,234] = 0.9, 0.4, 0.8, 0.6
    # I-405 & I-90
    A[274,246], A[274,82] = 1, 1
    D[274,246], D[274,82] = 0.8, 1.3
    A[85,276], A[85,112] = 1, 1 
    D[85,276], D[85,112] = 1.7, 0.8 
    A[114,246], A[114,82] = 1, 1 
    D[114,246], D[114,82] = 1.8, 2.1 
    A[243,112], A[243,276] = 1, 1
    D[243,112], D[243,276] = 1.2, 2.3
    
    return A, D

def Bi(x):
    return np.where(x > 0, 1, 0)

def GetFreeFlowReachableAdjacencyMatrix(A, D, delta_T = 1/60, ff_speed = 60.):
#     ff_speed = 60.
#     delta_T = 1/60  # 1 hour/60/3 --> 20 seconds
    ff_R= ff_speed * delta_T - D # free-flow reachable matrix
    ffRA = Bi(ff_R) # free-flow reachable adjacency matrix
    NonReachable_Idx = np.where(np.where(ffRA > 0, 1, 0) - A)
    
    WffRA = ffRA.copy() # weighted free-flow reachable adjacency matrix, ensure all adjacency nodes are reachable in one delta_t
    WffRA = ffRA.astype(np.float32)
    for idx in range(NonReachable_Idx[0].size):
        i = NonReachable_Idx[0][idx]
        j = NonReachable_Idx[1][idx]
        weight = 1.
        while ff_speed * delta_T * weight < D[i,j]:
            weight += 0.5
        WffRA[i,j] = weight
        
    
    return ffRA, WffRA

if __name__ == "__main__":
    speed_matrix_2015 =  pd.read_pickle('speed_matrix_2015')
    sensors = speed_matrix_2015.columns.values
    A = GetAdjacencyMatrix(sensors)