# MTGNN
This is a copy with some modification of the implementation that can be found here: https://github.com/nnzhan/MTGNN ,
which is a PyTorch implementation of the paper: [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650), published in KDD-2020.

## Requirements
The model is implemented using Python3.8.10 with dependencies specified in requirements.txt in bachelor directory

## Data Preparation

### Traffic datasets
Download data from the directory [data_MTGNN.zip]https://drive.google.com/file/d/1HkEvpZs_k-CqZQhN41BUqkt89mfDihPW/view?usp=sharing

Download the METR-LA and PEMS-BAY dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . Move them into the data folder. 



```
# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

## Model Training

### Multi-step


```
# METR-LA
python train_multi_step.py --adj_data ./data/sensor_graph/adj_mx.pkl --data ./data/METR-LA --num_nodes 207


# PEMS-BAY
python train_multi_step.py --adj_data ./data/sensor_graph/adj_mx_bay.pkl --data ./data/PEMS-BAY/ --num_nodes 325 --save ./PEMS-BAY
```

## Model Testing
* METR-LA
```
python test_multi_step.py --device cpu
```
* PEMS-BAY
```
python test_multi_step.py --device cpu --data data/PEMS-BAY --adj_data data/sensor_graph/adj_mx_bay.pkl --num_nodes 325 --checkpoint save/PEMS-BAY/exp1_0.pth --log_test log_MTGNN_test_bay.txt
```
