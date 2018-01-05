# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:38:57 2018

@author: zhiyong
"""


import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from Modules import FilterLinear

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
        


class BiLSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(BiLSTM, self).__init__()
        
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.il_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.fl_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.il_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl_b = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, input_f, input_b, Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b):
        
        combined_f = torch.cat((input_f, Hidden_State_f), 1)
        
        f_f = F.sigmoid(self.fl_f(combined_f))
        i_f = F.sigmoid(self.il_f(combined_f))
        o_f = F.sigmoid(self.ol_f(combined_f))
        C_f = F.tanh(self.Cl_f(combined_f))
        Cell_State_f = f_f * Cell_State_f + i_f * C_f
        Hidden_State_f = o_f * F.tanh(Cell_State_f)
        
        combined_b = torch.cat((input_b, Hidden_State_b), 1)

        f_b = F.sigmoid(self.fl_b(combined_b))
        i_b = F.sigmoid(self.il_b(combined_b))
        o_b = F.sigmoid(self.ol_b(combined_b))
        C_b = F.tanh(self.Cl_b(combined_b))
        Cell_State_b = f_b * Cell_State_b + i_b * C_b
        Hidden_State_b = o_b * F.tanh(Cell_State_b)
        
        return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b
    
    def loop(self, inputs):
        batch_size = inputs.size(0)
        steps = inputs.size(1)
        Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b = self.initHidden(batch_size)
        for i in range(steps):
            Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b = \
                self.forward(torch.squeeze(inputs[:,i:i+1,:]), torch.squeeze(inputs[:,steps-i-1:steps-i,:]), Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b)  
        return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State_f = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State_f = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Hidden_State_b = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State_b = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b
        else:
            Hidden_State_f = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State_f = Variable(torch.zeros(batch_size, self.hidden_size))
            Hidden_State_b = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State_b = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b
        
class GraphConvolutionalLSTM(nn.Module):
    
    def __init__(self, K, A, feature_size):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
        '''
        super(GraphConvolutionalLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        
        # here K = 3
        self.A1 = torch.Tensor(A)
        self.A2 = torch.matmul(self.A1, torch.Tensor(A))
        self.A3 = torch.matmul(self.A2, torch.Tensor(A))
#         self.filter_linear = nn.Linear(feature_size, feature_size)
        self.GC_1 = FilterLinear(feature_size, feature_size, self.A1)
        self.GC_2 = FilterLinear(feature_size, feature_size, self.A2)
        self.GC_3 = FilterLinear(feature_size, feature_size, self.A3)
        
        hidden_size = self.feature_size
        input_size = self.feature_size * K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, input, Hidden_State, Cell_State):
        
        x = input
        gc_1 = self.GC_1(x)
        gc_2 = self.GC_2(x)
        gc_3 = self.GC_3(x)
        gc = torch.cat((gc_1, gc_2, gc_3), 1)
            
        combined = torch.cat((gc, Hidden_State), 1)
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
        
        
        