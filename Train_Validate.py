# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:40:32 2018

@author: Zhiyong
"""
import torch
import numpy as np
from torch.autograd import Variable
import time
from Models import * 


def TrainRNN(train_dataloader, valid_dataloader, num_epochs = 3):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    rnn = RNN(input_dim, hidden_dim, output_dim)
    
    rnn.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(rnn.parameters(), lr = learning_rate)
    
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)

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
            
            loss_train = loss_MSE(outputs, labels)
            
            losses_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
            # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
            
            hidden = rnn.initHidden(batch_size)

            outputs = None
            for i in range(10):
                outputs, hidden = rnn(torch.squeeze(inputs_val[:,i:i+1,:]), hidden)

            loss_valid = loss_MSE(outputs, labels_val)
            losses_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(\
                                                                                         trained_number * batch_size, \
                                                                                         loss_interval_train,\
                                                                                         loss_interval_valid,\
                                                                                         np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

    return rnn, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]



def TrainLSTM(train_dataloader, valid_dataloader, num_epochs = 3):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    lstm = LSTM(input_dim, hidden_dim, output_dim)
    
    lstm.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(lstm.parameters(), lr = learning_rate)
    
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)

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

            loss_train = loss_MSE(Hidden_State, labels)
        
            losses_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            Hidden_State, Cell_State = lstm.loop(inputs_val)

            loss_valid = loss_MSE(Hidden_State, labels_val)
            losses_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(\
                                                                                         trained_number * batch_size, \
                                                                                         loss_interval_train,\
                                                                                         loss_interval_valid,\
                                                                                         np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

    return lstm, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]


def TrainGraphConvolutionalLSTM(train_dataloader, valid_dataloader, A, FFR, K, back_length = 3, num_epochs = 3, Clamp_A=False):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    gclstm = GraphConvolutionalLSTM(K, torch.Tensor(A), FFR[back_length], A.shape[0], Clamp_A=Clamp_A)
    
    gclstm.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(gclstm.parameters(), lr = learning_rate)
    
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0
        
        # validation data loader iterator init
        valid_dataloader_iter = iter(valid_dataloader)
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
                
            
            gclstm.zero_grad()

            Hidden_State, Cell_State = gclstm.loop(inputs)
            
            loss_train = loss_MSE(Hidden_State, labels)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
            losses_train.append(loss_train.data)
            
            # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            Hidden_State, Cell_State = gclstm.loop(inputs_val)
            loss_valid = loss_MSE(Hidden_State, labels)
            losses_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(\
                                                                                         trained_number * batch_size, \
                                                                                         loss_interval_train,\
                                                                                         loss_interval_valid,\
                                                                                         np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

    return gclstm, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]


def TrainGraphConvolutionalLSTM_Proposed(train_dataloader, valid_dataloader, A, FFR, K, back_length = 3, num_epochs = 3, Clamp_A=False, lambda_Aweight = 0.01, lambda_fea = 0.01):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    gclstm = GraphConvolutionalLSTM(K, torch.Tensor(A), FFR[back_length-1], A.shape[0], Clamp_A=Clamp_A)
    
    gclstm.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(gclstm.parameters(), lr = learning_rate)
    
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0
        
        # validation data loader iterator init
        valid_dataloader_iter = iter(valid_dataloader)
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
                
            gclstm.zero_grad()

            batch_size = inputs.size(0)
            time_step = inputs.size(1)
            Hidden_State, Cell_State = gclstm.initHidden(batch_size)
            
            previous_grads = []
            
            # Proposed Real Time Branching Learning Method
            for i in range(time_step):
                Hidden_State, Cell_State, gc = gclstm.forward(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)
                    
                gclstm.zero_grad()
                
                if i != time_step - 1: 
                    label_loss = loss_MSE(Hidden_State, torch.squeeze(inputs[:,i+1:i+2,:]))
                else:
                    label_loss = loss_MSE(Hidden_State, labels)

                # Graph Convolution Weight Regularization
                weight_loss = 0
                for idx in range(K):
                    gc_i_weight = gclstm.gc_list[idx].weight 
                    A_i = gclstm.A_list[idx]
                    weight_loss+=loss_L1(torch.mul(gc_i_weight, Variable(A_i).cuda()), target=torch.zeros_like(gc_i_weight))
                
                # Graph Convolution Features Regularization
                gc_loss = 0
                gc_features = torch.chunk(gc, K, 1)
                for idx in range(K-1):
                    gc_i = gc_features[idx]
                    gc_i1 = gc_features[idx+1]
                    gc_loss = gc_loss + loss_MSE(Variable(gc_i.data).cuda(), Variable(gc_i1.data).cuda())

                loss = label_loss + weight_loss * lambda_Aweight + gc_loss * lambda_Aweight

                optimizer.zero_grad()

                loss.backward()
                
                
                curr_grad = [x.grad.data for x in list(gclstm.parameters())] 
                
                if len(previous_grads) != 0: # not null
                    previous_grads_sum = previous_grads[0] # sum of gradients in previous steps 
                    for idx in range(1, len(previous_grads)):
                        pre_grad = previous_grads[idx]
                        previous_grads_sum += pre_grad
                    
                    # add previous gradients to current step only for the LSTM weights and bias, not for GC weights
                    idx = 0 
                    for x in list(gclstm.parameters()):
                        if idx >= K: # not add grads on GC1, GC2, GC3...
                            x.grad.data += previous_grads_sum[idx]
                        idx+=1

                # only store fixed steps of previous gradients ( length = back_length)
                if len(previous_grads) == back_length:
                    previous_grads.pop(0)
                    previous_grads.append(curr_grad)
                else:
                    previous_grads.append(curr_grad)

                optimizer.step()

                Hidden_State, Cell_State = gclstm.reinitHidden(batch_size, Hidden_State.data, Cell_State.data)
                    
            loss_train = loss_MSE(Hidden_State, labels)
            losses_train.append(loss_train.data)
            
            # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            Hidden_State, Cell_State = gclstm.loop(inputs_val)
            loss_valid = loss_MSE(Hidden_State, labels)
            losses_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
            
            
            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(\
                                                                                         trained_number * batch_size, \
                                                                                         loss_interval_train,\
                                                                                         loss_interval_valid,\
                                                                                         np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

    return gclstm, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]


def TestRNN(rnn, test_dataloader, max_speed):
    
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.MSELoss()
    
    tested_batch = 0
    
    losses_mse = []
    losses_l1 = [] 
    
    for data in test_dataloader:
        inputs, labels = data
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)

        # rnn.loop() 
        hidden = rnn.initHidden(batch_size)

        outputs = None
        for i in range(10):
            outputs, hidden = rnn(torch.squeeze(inputs[:,i:i+1,:]), hidden)
    
    
        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        loss_mse = loss_MSE(outputs, labels)
        loss_l1 = loss_L1(outputs, labels)
    
        losses_mse.append(loss_mse.data)
        losses_l1.append(loss_l1.data)
    
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    
    print('Tested: L1_mean: {}, L1_std : {}'.format(mean_l1, std_l1))
    return [losses_l1, losses_mse, mean_l1, std_l1]


def TestLSTM(lstm, test_dataloader, max_speed):
    
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.MSELoss()
    
    tested_batch = 0
    
    losses_mse = []
    losses_l1 = [] 
    
    for data in test_dataloader:
        inputs, labels = data
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)

        Hidden_State, Cell_State = lstm.loop(inputs)
    
        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        loss_mse = loss_MSE(Hidden_State, labels)
        loss_l1 = loss_L1(Hidden_State, labels)
    
        losses_mse.append(loss_mse.data)
        losses_l1.append(loss_l1.data)
    
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    
    print('Tested: L1_mean: {}, L1_std : {}'.format(mean_l1, std_l1))
    return [losses_l1, losses_mse, mean_l1, std_l1]


def TestGraphConvolutionalLSTM(gclstm, test_dataloader, max_speed):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    tested_batch = 0
    
    losses_mse = []
    losses_l1 = [] 
    
    for data in test_dataloader:
        inputs, labels = data
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)

        Hidden_State, Cell_State = gclstm.loop(inputs)
    
        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        loss_mse = loss_MSE(Hidden_State, labels)
        loss_l1 = loss_L1(Hidden_State, labels)
    
        losses_mse.append(loss_mse.data)
        losses_l1.append(loss_l1.data)
    
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    
    print('Tested: L1_mean: {}, L1_std : {}'.format(mean_l1, std_l1))
    return [losses_l1, losses_mse, mean_l1, std_l1]
