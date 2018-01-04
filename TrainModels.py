# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:16:33 2018

@author: zhiyong
"""

import torch.utils.data as utils
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
from Models import * 
import time

speed_matrix_2015 =  pd.read_pickle('speed_matrix_2015')

speed_matrix = speed_matrix_2015


def PrepareDataset(speed_matrix, BATCH_SIZE = 10, seq_len = 10, pred_len = 1, train_propotion = 0.8):
    """ Prepare training and testing datasets and dataloaders.
    
    Convert speed/volume/occupancy matrix to training and testing dataset. 
    The vertical axis of speed_matrix is the time axis and the horizontal axis 
    is the spatial axis.
    
    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    time_len = speed_matrix.shape[0]
    
    max_speed = speed_matrix.max().max()
    speed_matrix =  speed_matrix / max_speed
    
    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i+seq_len].values)
        speed_labels.append(speed_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)
    
    # shuffle and split the dataset to training and testing datasets
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype = int)
    np.random.shuffle(index)
    split_index = int(np.floor(sample_size * train_propotion))
    
    train_data, train_label = speed_sequences[:split_index], speed_labels[:split_index]
    test_data, test_label = speed_sequences[split_index:], speed_labels[split_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    train_dataset = utils.TensorDataset(train_data, train_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True)
    
    return train_dataloader, test_dataloader


def TrainRNN(train_dataloader, num_epochs = 3):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    rnn = RNN(input_dim, hidden_dim, output_dim)
    
    rnn.cuda()
    
    loss_fn = torch.nn.MSELoss()

    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(rnn.parameters(), lr = learning_rate)
    
    use_gpu = torch.cuda.is_available()
    
    losses = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
                
            rnn.zero_grad()
            
            # rnn.loop() 
            hidden = rnn.initHidden(batch_size)

            outputs = None
            for i in range(10):
                outputs, hidden = rnn(torch.squeeze(inputs[:,i:i+1,:]), hidden)
            #######
            
            loss = loss_fn(outputs, labels)
        
            losses.append(loss.data)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            trained_number += 1

            if trained_number % 1000 == 0:
                cur_time = time.time()
                print('trained #: {}, loss: {}, time: {}'.format( \
                      trained_number * batch_size, \
                      np.around([loss.data[0]], decimals=8), \
                      np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

    return rnn, losses

def TrainLSTM(train_dataloader, num_epochs = 3):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    lstm = LSTM(input_dim, hidden_dim, output_dim)
    
    lstm.cuda()
    
    loss_fn = torch.nn.MSELoss()

    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(lstm.parameters(), lr = learning_rate)
    
    use_gpu = torch.cuda.is_available()
    
    losses = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
                
            lstm.zero_grad()

            Hidden_State, Cell_State = lstm.loop(inputs)

            loss = loss_fn(Hidden_State, labels)
        
            losses.append(loss.data)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            trained_number += 1

            if trained_number % 1000 == 0:
                cur_time = time.time()
                print('trained #: {}, loss: {}, time: {}'.format( \
                      trained_number * batch_size, \
                      np.around([loss.data[0]], decimals=8), \
                      np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

    return lstm, losses

if __name__ == "__main__":
    train_dataloader, test_dataloader = PrepareDataset(speed_matrix)
    rnn, losses = TrainRNN(train_dataloader, num_epochs = 10)
    lstm, losses = TrainLSTM(train_dataloader, num_epochs = 10)
    