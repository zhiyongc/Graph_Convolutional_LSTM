### Traffic Graph Convolutional Recurrent Neural Network
### A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting
------
Extension version of **High-order Graph Convolutional Recurrent Neural Network**

### 

### Environment
* Python 3.6.1
* PyTorch 0.3.0

### Model Structure
![alt text](/Images/HGC-LSTM2.tiff)
![alt text](/Images/HGC-LSTM.png)


### Results
The results can be found in the [WiKi](https://github.com/zhiyongc/GraphConvolutionalLSTM/wiki)

## Data 
To run the code, you need to download the loop detector data from my GitHub link: https://github.com/zhiyongc/Seattle-Loop-Data and put them in the "Data" folder. For confidentiality, the INRIX data can not be shared.

<!--


Description of the datasets:
* `inrix_seattle_speed_matrix_2012`: INRIX Speed Matrix (read by Pandas)
* `INRIX_Seattle_2012_A.npy`: INRIX Adjacency Matrix
* `INRIX_Seattle_2012_reachability_free_flow_Xmin.npy`: INRIX Free-flow Reachability Matrix during X minites' drive
* `nodes_inrix_tmc_list.csv`: List of INRIX TMC code, with the same order of that in the INRIX Speed Matrix (not needed to run the code)
* `speed_matrix_2015`: Loop Speed Matrix
* `Loop_Seattle_2015_A.npy`: Loop Adjacency Matrix
* `Loop_Seattle_2015_reachability_free_flow_5min.npy`: Loop Free-flow Reachability Matrix during X minites' drive
* `nodes_loop_mp_list.csv`: List of loop detectors' milepost, with the same order of that in the Loop Speed Matrix (not needed to run the code)
-->
## Cite
Please cite our paper if you use this code or data in your own work:
[High-Order Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting](https://arxiv.org/abs/1802.07007)
```
@misc{1802.07007,
  Author = {Zhiyong Cui and Kristian Henrickson and Ruimin Ke and Yinhai Wang},
  Title = {High-Order Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting},
  Year = {2018},
  Eprint = {arXiv:1802.07007},
}
```


