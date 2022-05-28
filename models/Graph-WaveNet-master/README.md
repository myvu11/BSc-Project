# Graph WaveNet for Deep Spatial-Temporal Graph Modeling
This is a copy with some modification of the implementation that can be found here: https://github.com/nnzhan/Graph-WaveNet
That is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).


<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

## Requirements
The model is implemented using Python3.8.10 with dependencies specified in requirements.txt in bachelor directory


## Data Preparation

### Step1: Download data files
Download data from the directory [data_GraphWaveNet.zip](https://drive.google.com/file/d/1Hmv66EffxqDbqSM4udAmitKuzoxA-SIg/view?usp=sharing)



### Step2: Process raw data 

```
# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
## Train Commands

```
# METR-LA
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj

# PEMS-BAY
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --num_nodes 325 --save ./garage/pems --log_train log_Graph_train_bay.txt

```

## Test Commands

```
# METR-LA
python test.py --device cpu

# PEMS-BAY
python test.py --device cpu --data data/PEMS-BAY/ --adjdata data/sensor_graph/adj_mx_bay.pkl --num_nodes 325 --checkpoint ./garage/pems_exp1_best_1.6.pth --log_test log_Graph_test_bay.txt

```