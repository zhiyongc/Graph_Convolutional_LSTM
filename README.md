# High-Order Graph Convolutional Recurrent Neural Network
## A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting

###

### Environment
* Python 3.6.1
* PyTorch 0.3.0

### Model Structure
![alt text](/Images/HGC-LSTM.png)


### Results
The results can be found in the [WiKi](https://github.com/zhiyongc/GraphConvolutionalLSTM/wiki)

### Data 
To run the code, you need to download the data from the following link: https://drive.google.com/drive/folders/1Mw8tjiPD-wknFu6dY5NTw4tqOiu5X9rz?usp=sharing and put them in the "Data" folder. The data contains two traffic networks in Seattle: a loop detector based freeway network and an INRIX data-based urban traffic network. The details about these netowrk is described in the reference paper.
Description of data
* `inrix_seattle_speed_matrix_2012`: INRIX Speed Matrix (read by Pandas)
* `INRIX_Seattle_2012_A.npy`: INRIX Adjacency Matrix
* `INRIX_Seattle_2012_reachability_free_flow_Xmin.npy`: INRIX Free-flow Reachability Matrix during X minites' drive
* `speed_matrix_2015`: Loop Speed Matrix
* `Loop_Seattle_2015_A.npy`: Loop Adjacency Matrix
* `Loop_Seattle_2015_reachability_free_flow_5min.npy`: Loop Free-flow Reachability Matrix during X minites' drive


### Cite
[High-Order Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting](https://arxiv.org/abs/1802.07007)
```
{
}
```


