### Traffic Graph Convolutional Recurrent Neural Network
### A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting
------
##### Extended version of *High-order Graph Convolutional Recurrent Neural Network*



### 2nd version of the TGC-LSTM Model Structure

![alt text](/Images/HGC-LSTM2.png)

* The 2nd version of the structure of Traffic Graph Convolutional LSTM (TGC-LSTM). 
* The traffic graph convolution module is designed based on the physical network topology.
* The code of this model is in the Code_V2 folder.
  * Environment (Jupyter Notebook): Python 3.6.1 and PyTorch 0.4.1
  * The code contains the implementations and results of the compared models, including LSTM, spectral graph convolution LSTM, localized spectral graph convolution LSTM.

------

### 1st version of the High-order Graph Convolutional Recurrent Neural Network Structure 

<img src="/Images/HGC-LSTM.png" alt="drawing" width="800"/>

* The 1st version of Traffic Graph Convolutional LSTM. 
* The code of this model is in the Code_V1 folder.
  * Environment: Python 3.6.1 and PyTorch 0.3.0
  
------

### Experimental Results 
###### Validation Loss Comparison Chart
<img src="/Images/V2_Validation_loss.png" alt="drawing" width="400"/>

For more detailed experimental results, please refer to [our paper](https://arxiv.org/abs/1802.07007).
<!-- The results can be found in the [WiKi](https://github.com/zhiyongc/GraphConvolutionalLSTM/wiki) --->

------

### Data 
To run the code, you need to download the loop detector data and the network topology information from the link: https://github.com/zhiyongc/Seattle-Loop-Data and put them in the "Data" folder. 

For confidentiality, the INRIX data can not be shared.

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

------


### Updated Citation
Please cite our paper if you use this code or data in your own work:
[Traffic Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting](https://arxiv.org/abs/1802.07007) 

Hope our work is benefitial for you. Thanks!
```
@article{cui2018high,
  title={High-Order Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting},
  author={Cui, Zhiyong and Henrickson, Kristian and Ke, Ruimin and Wang, Yinhai},
  journal={arXiv preprint arXiv:1802.07007},
  year={2018}
}
```


