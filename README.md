### Traffic Graph Convolutional Recurrent Neural Network
### A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting
------
##### Extended version of *High-order Graph Convolutional Recurrent Neural Network*



### 2nd version of the TGC-LSTM Model Structure

![alt text](/Images/TGC-LSTM.png)

* The 2nd version of the structure of Traffic Graph Convolutional LSTM (TGC-LSTM). 
  * ![equation](http://mathurl.com/y9brdy6u.png) is the K-th order adjacency matrix
  * ![equation](http://mathurl.com/y6w9d7bj.png) is the Free Flow Reachability matrix defined based on the network physical topology information.
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
### Dataset
The model is tested on two real-world network-wide traffic speed dataset, loop detector data and INRIX data. The following figure shows the covered areas. (a) Seattle freeway network; (b) Seattle downtown roadway network.

<img src="/Images/dataset.png" alt="drawing" width="400"/>

Check out this [Link](https://github.com/zhiyongc/Seattle-Loop-Data) for looking into and downloading the **loop detecotr dataset**. For confidentiality reasons, the **INRIX dataset** can not be shared.

To run the code, you need to download the loop detector data and the network topology information and put them in the proper "Data" folder. 

------

### Experimental Results 
###### Validation Loss Comparison Chart & Model Performance with respect to the number of K
<img src="/Images/V2_Validation_loss.png" alt="drawing" width="400"/><img src="/Images/K_results.png" alt="drawing" width="350"/>

For more detailed experimental results, please refer to [the paper](https://arxiv.org/abs/1802.07007).
<!-- The results can be found in the [WiKi](https://github.com/zhiyongc/GraphConvolutionalLSTM/wiki) --->
------

### Visualization
###### Visualization of graph convolution (GC) weight matrices (averaged, K=3) & weight values on real maps
<img src="/Images/weight_matrix.png" alt="drawing" width="600"/>
<img src="/Images/visualization.png" alt="drawing" width="680"/>

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
[Traffic Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting](https://ieeexplore.ieee.org/abstract/document/8917706) 

Hope our work is benefitial for you. Thanks!
```
@article{cui2019traffic,
  title={Traffic graph convolutional recurrent neural network: A deep learning framework for network-scale traffic learning and forecasting},
  author={Cui, Zhiyong and Henrickson, Kristian and Ke, Ruimin and Wang, Yinhai},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}
```


