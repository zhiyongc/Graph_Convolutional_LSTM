# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:38:57 2018

@author: zhiyong
"""


import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from Modules import FilterLinear
import math
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
#         print(combined)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            return Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(batch_size, self.hidden_size))
        
        
class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM, self).__init__()
        
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State = self.forward(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)  
        return Hidden_State, Cell_State
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State
        
        
class GraphConvolutionalLSTM(nn.Module):
    
    def __init__(self, K, A, FFR, feature_size, Clamp_A=True):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            FFR: free-flow reachability matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(GraphConvolutionalLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        
        self.K = K
        
        self.A_list = [] # Adjacency Matrix List
        A = torch.FloatTensor(A)
        A_temp = torch.eye(feature_size,feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, torch.Tensor(A))
            if Clamp_A:
                # confine elements of A 
                A_temp = torch.clamp(A_temp, max = 1.) 
            self.A_list.append(torch.mul(A_temp, torch.Tensor(FFR)))
#             self.A_list.append(A_temp)
        
        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList([FilterLinear(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])                  
        
        hidden_size = self.feature_size
        input_size = self.feature_size * K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input, Hidden_State, Cell_State):
        
        x = input

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)
            
        combined = torch.cat((gc, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,  torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State, gc
    
    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a
    
    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State, gc = self.forward(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)  
        return Hidden_State, Cell_State
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State
    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            return Hidden_State, Cell_State