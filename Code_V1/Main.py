# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:46:06 2018

@author: Zhiyong
"""
import torch.utils.data as utils
import torch
import numpy as np
import pandas as pd
from Models import * 
from Train_Validate import * 



def PrepareDataset(speed_matrix, BATCH_SIZE = 40, seq_len = 10, pred_len = 1, train_propotion = 0.7, valid_propotion = 0.2):
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
    
    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * ( train_propotion + valid_propotion)))
    
    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    
    return train_dataloader, valid_dataloader, test_dataloader, max_speed


if __name__ == "__main__":
    data = 'inrix'
#    data = 'loop'
    if data == 'inrix':
        speed_matrix =  pd.read_pickle('../Data/inrix_seattle_speed_matrix_2012')
        A = np.load('../Data/INRIX_Seattle_2012_A.npy')
        FFR_5min = np.load('../Data/INRIX_Seattle_2012_reachability_free_flow_5min.npy')
        FFR_10min = np.load('../Data/INRIX_Seattle_2012_reachability_free_flow_10min.npy')
        FFR_15min = np.load('../Data/INRIX_Seattle_2012_reachability_free_flow_15min.npy')
        FFR_20min = np.load('../Data/INRIX_Seattle_2012_reachability_free_flow_20min.npy')
        FFR_25min = np.load('../Data/INRIX_Seattle_2012_reachability_free_flow_25min.npy')
        FFR = [FFR_5min, FFR_10min, FFR_15min, FFR_20min, FFR_25min]
    elif data == 'loop':
        speed_matrix =  pd.read_pickle('../Data/speed_matrix_2015')
        A = np.load('../Data/Loop_Seattle_2015_A.npy')
        FFR_5min = np.load('../Data/Loop_Seattle_2015_reachability_free_flow_5min.npy')
        FFR_10min = np.load('../Data/Loop_Seattle_2015_reachability_free_flow_10min.npy')
        FFR_15min = np.load('../Data/Loop_Seattle_2015_reachability_free_flow_15min.npy')
        FFR_20min = np.load('../Data/Loop_Seattle_2015_reachability_free_flow_20min.npy')
        FFR_25min = np.load('../Data/Loop_Seattle_2015_reachability_free_flow_25min.npy')
        FFR = [FFR_5min, FFR_10min, FFR_15min, FFR_20min, FFR_25min]
#        
#    train_dataloader, valid_dataloader, test_dataloader, max_speed = PrepareDataset(speed_matrix)
#
    rnn, rnn_loss = TrainRNN(train_dataloader, valid_dataloader, num_epochs = 1)
    ### rnn_loss = [losses_train, losses_interval_train, losses_valid, losses_interval_valid]
    rnn_test = TestRNN(rnn, test_dataloader, max_speed ) 
    ### rnn_test = [losses_l1, losses_mse, mean_l1, std_l1]
    
    lstm, lstm_loss = TrainLSTM(train_dataloader, valid_dataloader, num_epochs = 1)
    lstm_test = TestLSTM(lstm, test_dataloader, max_speed )
    
    gclstm, gclstm_loss = TrainGraphConvolutionalLSTM(train_dataloader, valid_dataloader, A, FFR, K=3, back_length = 2, num_epochs = 1, Clamp_A = True)
    gclstm_test = TestGraphConvolutionalLSTM(gclstm, test_dataloader, max_speed)
        
    gclstm_proposed, gclstm_proposed_loss = TrainGraphConvolutionalLSTM_Proposed(train_dataloader, valid_dataloader, A, FFR, K=3, back_length = 2, num_epochs = 1, Clamp_A = True, lambda_Aweight = 0.01, lambda_fea = 0.01)
    gclstm_proposed_test = TestGraphConvolutionalLSTM(gclstm_proposed, test_dataloader, max_speed)

        
        
        
        
        