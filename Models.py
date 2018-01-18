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
        


class AdaptiveGraphConvolutionalLSTM(nn.Module):
    
    def __init__(self, K, batch_size, step_size, feature_size):
        '''
        Args:
            K: K-hop graph
            A: Directed adjacency matrix
            WffRA: Weighted free-flow reachable matrix
            WffRA_I: WffRA - I
            max_speed: max speed value in the dataset, which is used to calculate true speed from the normalized speed data
            delta_T: time interval of one step in the dataset
            
        '''
        super(AdaptiveGraphConvolutionalLSTM, self).__init__()
        self.batch_size = batch_size
        self.step_size = step_size
        self.feature_size = feature_size
        self.hidden_size = feature_size
        
        self.K = K
        self.A = None
        self.D = None
        self.WffRA = None
        self.WffRA_I = None
        self.max_speed = None
        self.delta_T = None
        self.I = None
        # here K = 3
#         self.A1 = torch.Tensor(A)
#         self.A2 = torch.matmul(self.A1, torch.Tensor(A))
#         self.A3 = torch.matmul(self.A2, torch.Tensor(A))

#         self.filter_linear = nn.Linear(feature_size, feature_size)
#         self.GC_1 = FilterLinear(feature_size, feature_size)
#         self.GC_2 = FilterLinear(feature_size, feature_size)
#         self.GC_3 = FilterLinear(feature_size, feature_size)
        self.GC_R_weight = Parameter(torch.Tensor(self.feature_size, self.feature_size).cuda())
        stdv = 1. / math.sqrt(self.GC_R_weight.size(1))
        self.GC_R_weight.data.uniform_(-stdv, stdv)
        self.GC_NR_weight = Parameter(torch.Tensor(self.feature_size, self.feature_size).cuda())
        stdv = 1. / math.sqrt(self.GC_NR_weight.size(1))
        self.GC_NR_weight.data.uniform_(-stdv, stdv)
#         self.GC_R = AdaptiveFilterLinear(feature_size, feature_size)
#         self.GC_NR = AdaptiveFilterLinear(feature_size, feature_size)
        
        hidden_size = self.feature_size
        input_size = self.feature_size * K * 2

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, input, filters, Hidden_State, Cell_State):
        
        (R1_i, R2_i, R3_i, NR1_i, NR2_i, NR3_i) = filters
        
        x = input.unsqueeze(1)
#         weight = torch.mul(self.GC_R_weight, Variable(R1_i.cuda(), requires_grad=False))
        
        gc_R1 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_R_weight, Variable(R1_i.cuda(), requires_grad=False)))) # gc_R1 = x . W \odot R1_i
        gc_R2 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_R_weight, Variable(R2_i.cuda(), requires_grad=False))))
        gc_R3 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_R_weight, Variable(R3_i.cuda(), requires_grad=False))))
        gc_R = torch.cat((gc_R1, gc_R2, gc_R3), 1)
#         gc_R1 = self.GC_R(x, R1_i)
#         gc_R2 = self.GC_R(x, R2_i)
#         gc_R3 = self.GC_R(x, R3_i)
        
        gc_NR1 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_NR_weight, Variable(R1_i.cuda(), requires_grad=False)))) # gc_NR1 = x . W \odot NR1_i
        gc_NR2 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_NR_weight, Variable(R2_i.cuda(), requires_grad=False)))) 
        gc_NR3 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_NR_weight, Variable(R3_i.cuda(), requires_grad=False))))
        gc_NR= torch.cat((gc_NR1, gc_NR2, gc_NR3), 1)
#         gc_NR1 = self.GC_NR(x, NR1_i)
#         gc_NR2 = self.GC_NR(x, NR2_i)
#         gc_NR3 = self.GC_NR(x, NR3_i)
        
            
        combined = torch.cat((gc_R, gc_NR, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a
    
    def initReachableMatrixTensors(self, A, D, WffRA, max_speed, delta_T):
        self.I = torch.cuda.FloatTensor(np.identity(self.feature_size))
        self.WffRA = torch.cuda.FloatTensor(WffRA)
        self.WffRA_I = self.WffRA - self.I 
        self.A = torch.cuda.FloatTensor(A)
        self.D = torch.cuda.FloatTensor(D)
        self.max_speed = float(max_speed)
        self.delta_T = float(delta_T)
        
    def getAdaptiveReacheableMatrix(self, inputs):
        [batch_size, step_size, feature_size] = inputs.size()
        
        A1 = self.A - self.I
        A2 = torch.mm(A1, A1)
        A3 = torch.mm(A2, A1)
        
#         print(self.I.shape, self.A.shape, self.D.shape, self.WffRA_I.shape)
        
        I = self.I.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        A = self.A.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        D = self.D.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        WffRA_I = self.WffRA_I.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        A1 = A1.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        A2 = A2.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        A3 = A3.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        

        R_1 = [] # list of 1-hop reachable matrix
        NR_1 = [] # list of 1-hop non-reachable matrix
        for i in range(step_size):
            vi = torch.squeeze(inputs[:,i:i+1,:])
            vi_seq = []
            for i in range(vi.shape[0]):
                vi_seq.append(torch.diag(vi[i]))
            vi = torch.stack(vi_seq) # vi is Variable here
            Vi = vi.data.bmm(I) # bmm: batch matrix multiplication (vi.data convert it to tensor)
            ViA = Vi.bmm(WffRA_I)
            R1i = self.Bi_torch(ViA * self.max_speed * self.delta_T - D)
            R_1.append(R1i)
            NR1i = A1 - R1i
            NR_1.append(NR1i)
        
        R_2 = [] # list of 2-hop reachable matrix
        NR_2 = [] # list of 2-hop non-reachable matrix
        for i in range(step_size-1):
            R2i = self.Bi_torch(R_1[i].bmm(R_1[i+1]))
            R_2.append(R2i)
            NR2i = A2 - R2i
            NR_2.append(NR2i)

        R_3 = [] # list of 3-hop reachable matrix
        NR_3 = [] # list of 3-hop non-reachable matrix
        for i in range(step_size-2):
            R3i = self.Bi_torch(R_1[i].bmm(R_2[i+1]))
            R_3.append(R3i)
            NR3i = A3 - R3i
            NR_3.append(NR3i)
        return R_1, R_2, R_3, NR_1, NR_2, NR_3
    
    def loop(self, inputs, A, D, WffRA, max_speed, delta_T):
        
        R1, R2, R3, NR1, NR2, NR3 = self.getAdaptiveReacheableMatrix(inputs)
        
#         print('ARM finished')
        
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        
        for i in range(time_step):
            R1i = R1[i]
            NR1i = NR1[i]
            R2i = torch.zeros_like(R1i)
            NR2i = torch.zeros_like(R1i)
            R3i = torch.zeros_like(R1i)
            NR3i = torch.zeros_like(R1i)
            if i-1 >= 0 :
                R2i = R2[i-1]
                NR2i = NR2[i-1]
            if i-2 >= 0:
                R3i = R3[i-2]
                NR3i = NR3[i-2]
            
            filters = (R1i, R2i, R3i, NR1i, NR2i, NR3i)
            
            Hidden_State, Cell_State = self.forward(torch.squeeze(inputs[:,i:i+1,:]), filters, Hidden_State, Cell_State)  
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


class AdaptiveGraphConvolutionalLSTM_MW(nn.Module):
    '''
        MW: multiple weight matrices, namely each hop graph convolution has a weight matrix
    '''
    
    def __init__(self, K, batch_size, step_size, feature_size):
        '''
        Args:
            K: K-hop graph
            A: Directed adjacency matrix
            WffRA: Weighted free-flow reachable matrix
            WffRA_I: WffRA - I
            max_speed: max speed value in the dataset, which is used to calculate true speed from the normalized speed data
            delta_T: time interval of one step in the dataset
            
        '''
        super(AdaptiveGraphConvolutionalLSTM_MW, self).__init__()
        self.batch_size = batch_size
        self.step_size = step_size
        self.feature_size = feature_size
        self.hidden_size = feature_size
        
        
        self.A = None
        self.D = None
        self.WffRA = None
        self.WffRA_I = None
        self.max_speed = None
        self.delta_T = None
        self.I = None
        # here K = 3
#         self.A1 = torch.Tensor(A)
#         self.A2 = torch.matmul(self.A1, torch.Tensor(A))
#         self.A3 = torch.matmul(self.A2, torch.Tensor(A))

#         self.filter_linear = nn.Linear(feature_size, feature_size)
#         self.GC_1 = FilterLinear(feature_size, feature_size)
#         self.GC_2 = FilterLinear(feature_size, feature_size)
#         self.GC_3 = FilterLinear(feature_size, feature_size)
#         self.GC_R = AdaptiveFilterLinear(feature_size, feature_size)
#         self.GC_NR = AdaptiveFilterLinear(feature_size, feature_size)
        self.GC_R1_weight = Parameter(torch.Tensor(self.feature_size, self.feature_size).cuda())
        stdv = 1. / math.sqrt(self.GC_R1_weight.size(1))
        self.GC_R1_weight.data.uniform_(-stdv, stdv)
        self.GC_NR1_weight = Parameter(torch.Tensor(self.feature_size, self.feature_size).cuda())
        stdv = 1. / math.sqrt(self.GC_NR1_weight.size(1))
        self.GC_NR1_weight.data.uniform_(-stdv, stdv)
        
        self.GC_R2_weight = Parameter(torch.Tensor(self.feature_size, self.feature_size).cuda())
        stdv = 1. / math.sqrt(self.GC_R2_weight.size(1))
        self.GC_R2_weight.data.uniform_(-stdv, stdv)
        self.GC_NR2_weight = Parameter(torch.Tensor(self.feature_size, self.feature_size).cuda())
        stdv = 1. / math.sqrt(self.GC_NR2_weight.size(1))
        self.GC_NR2_weight.data.uniform_(-stdv, stdv)
        
        self.GC_R3_weight = Parameter(torch.Tensor(self.feature_size, self.feature_size).cuda())
        stdv = 1. / math.sqrt(self.GC_R3_weight.size(1))
        self.GC_R3_weight.data.uniform_(-stdv, stdv)
        self.GC_NR3_weight = Parameter(torch.Tensor(self.feature_size, self.feature_size).cuda())
        stdv = 1. / math.sqrt(self.GC_NR3_weight.size(1))
        self.GC_NR3_weight.data.uniform_(-stdv, stdv)

        
        hidden_size = self.feature_size
        input_size = self.feature_size * K * 2

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, input, filters, Hidden_State, Cell_State):
        
        (R1_i, R2_i, R3_i, NR1_i, NR2_i, NR3_i) = filters
        
        x = input.unsqueeze(1)
#         weight = torch.mul(self.GC_R_weight, Variable(R1_i.cuda(), requires_grad=False))
        
        gc_R1 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_R1_weight, Variable(R1_i.cuda(), requires_grad=False)))) # gc_R1 = x . W \odot R1_i
        gc_R2 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_R2_weight, Variable(R2_i.cuda(), requires_grad=False))))
        gc_R3 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_R3_weight, Variable(R3_i.cuda(), requires_grad=False))))
        gc_R = torch.cat((gc_R1, gc_R2, gc_R3), 1)
#         gc_R1 = self.GC_R(x, R1_i)
#         gc_R2 = self.GC_R(x, R2_i)
#         gc_R3 = self.GC_R(x, R3_i)
        
        gc_NR1 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_NR1_weight, Variable(R1_i.cuda(), requires_grad=False)))) # gc_NR1 = x . W \odot NR1_i
        gc_NR2 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_NR2_weight, Variable(R2_i.cuda(), requires_grad=False)))) 
        gc_NR3 = torch.squeeze(torch.bmm(x, torch.mul(self.GC_NR3_weight, Variable(R3_i.cuda(), requires_grad=False))))
        gc_NR= torch.cat((gc_NR1, gc_NR2, gc_NR3), 1)
#         gc_NR1 = self.GC_NR(x, NR1_i)
#         gc_NR2 = self.GC_NR(x, NR2_i)
#         gc_NR3 = self.GC_NR(x, NR3_i)
        
            
        combined = torch.cat((gc_R, gc_NR, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a
    
    def initReachableMatrixTensors(self, A, D, WffRA, max_speed, delta_T):
        self.I = torch.cuda.FloatTensor(np.identity(self.feature_size))
        self.WffRA = torch.cuda.FloatTensor(WffRA)
        self.WffRA_I = self.WffRA - self.I 
        self.A = torch.cuda.FloatTensor(A)
        self.D = torch.cuda.FloatTensor(D)
        self.max_speed = float(max_speed)
        self.delta_T = float(delta_T)
        
    def getAdaptiveReacheableMatrix(self, inputs):
        [batch_size, step_size, feature_size] = inputs.size()
        
        A1 = self.A 
#         A1 = self.A - self.I
        A2 = torch.mm(A1, A1)
        A3 = torch.mm(A2, A1)
        
#         print(self.I.shape, self.A.shape, self.D.shape, self.WffRA_I.shape)
        
        I = self.I.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        A = self.A.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        D = self.D.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        WffRA = self.WffRA.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        WffRA_I = self.WffRA_I.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        A1 = A1.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        A2 = A2.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        A3 = A3.unsqueeze(0).expand(batch_size, feature_size, feature_size)
        

        R_1 = [] # list of 1-hop reachable matrix
        NR_1 = [] # list of 1-hop non-reachable matrix
        for i in range(step_size):
            vi = torch.squeeze(inputs[:,i:i+1,:])
            vi_seq = []
            for i in range(vi.shape[0]):
                vi_seq.append(torch.diag(vi[i]))
            vi = torch.stack(vi_seq) # vi is Variable here
            Vi = vi.data.bmm(I) # bmm: batch matrix multiplication (vi.data convert it to tensor)
            ViA = Vi.bmm(WffRA)
#             ViA = Vi.bmm(WffRA_I)
            R1i = self.Bi_torch(ViA * self.max_speed * self.delta_T - D)
            R_1.append(R1i)
            NR1i = A1 - R1i
            NR_1.append(NR1i)
        
        R_2 = [] # list of 2-hop reachable matrix
        NR_2 = [] # list of 2-hop non-reachable matrix
        for i in range(step_size-1):
            R2i = self.Bi_torch(R_1[i].bmm(R_1[i+1]))
            R_2.append(R2i)
            NR2i = A2 - R2i
            NR_2.append(NR2i)

        R_3 = [] # list of 3-hop reachable matrix
        NR_3 = [] # list of 3-hop non-reachable matrix
        for i in range(step_size-2):
            R3i = self.Bi_torch(R_1[i].bmm(R_2[i+1]))
            R_3.append(R3i)
            NR3i = A3 - R3i
            NR_3.append(NR3i)
        return R_1, R_2, R_3, NR_1, NR_2, NR_3
    
    def loop(self, inputs, A, D, WffRA, max_speed, delta_T):
        
        R1, R2, R3, NR1, NR2, NR3 = self.getAdaptiveReacheableMatrix(inputs)
        
#         print('ARM finished')
        
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        
        for i in range(time_step):
            R1i = R1[i]
            NR1i = NR1[i]
            R2i = torch.zeros_like(R1i)
            NR2i = torch.zeros_like(R1i)
            R3i = torch.zeros_like(R1i)
            NR3i = torch.zeros_like(R1i)
            if i-1 >= 0 :
                R2i = R2[i-1]
                NR2i = NR2[i-1]
            if i-2 >= 0:
                R3i = R3[i-2]
                NR3i = NR3[i-2]
            
            filters = (R1i, R2i, R3i, NR1i, NR2i, NR3i)
            
            Hidden_State, Cell_State = self.forward(torch.squeeze(inputs[:,i:i+1,:]), filters, Hidden_State, Cell_State)  
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