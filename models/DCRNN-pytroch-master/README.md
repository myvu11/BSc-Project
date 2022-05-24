# 
This is a copy with some modification of the implementation that can be found here: https://github.com/KimMeen/DCRNN

# DCRNN

A simplified PyTorch implementation of the paper https://arxiv.org/abs/1707.01926 based on [xlwang233/pytorch-DCRNN](https://github.com/xlwang233/pytorch-DCRNN) and [chnsh/DCRNN_PyTorch](https://github.com/chnsh/DCRNN_PyTorch)

The original tensorflow implementation: [liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)

Welcome to open issues for any problems/questions!

## Requirements

The model is implemented using Python3.8.10 with dependencies specified in requirements.txt in bachelor directory

## Training

Firstly, you need to pre-process the data by using ```generate_dataset.py```:

```bash
# METR-LA
python -m generate_dataset --output_dir=data/processed/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m generate_dataset --output_dir=data/processed/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

 Then execute the training script (default setup on METR-LA):

```bash
python train.py
```
 
 For training on PEMS-BAY 
```bash
python train.py --data data/PEMS-BAY --num_nodes 325 --checkpoints ./checkpoints/PEMS-BAY/dcrnn.pt --sensor_ids ./data/sensor_graph/graph_sensor_ids_bay.txt --sensor_distance ./data/sensor_graph/distances_bay_2017.csv --recording data/processed/PEMS-BAY
```

## Testing

Run single-step testing on trained model from horizon 1 to 12:

```bash
python single_step_test.py
```

Run multi-step testing on trained model from horizon 1 to 12:

Test on METR-LA:
```bash
python multi_step_test.py
```


Test on PEMS-BAY:
```bash
python multi_step_test.py --num_nodes 325 --checkpoints ./checkpoints/PEMS-BAY/dcrnn.pt --sensor_ids ./data/sensor_graph/graph_sensor_ids_bay.txt --sensor_distance ./data/sensor_graph/distances_bay_2017.csv --recording data/processed/PEMS-BAY --log_test log_DCRNN_test_bay.txt
```

P.S. Single-step testing evaluate on `y_true[:, horizon_i, :, :] ` and `y_pred[:, horizon_i, :, :]` while multi-step testing based on `y_true[:, :horizon_i, :, :] ` and `y_pred[:, :horizon_i, :, :]` 

## Results

Single-step testing results on 80 epochs:

```
--------single-step testing results--------
Horizon 01, MAE: 2.21, MAPE: 0.0526, RMSE: 3.80
Horizon 02, MAE: 2.51, MAPE: 0.0624, RMSE: 4.64
Horizon 03, MAE: 2.71, MAPE: 0.0698, RMSE: 5.21
Horizon 04, MAE: 2.88, MAPE: 0.0760, RMSE: 5.68
Horizon 05, MAE: 3.02, MAPE: 0.0813, RMSE: 6.04
Horizon 06, MAE: 3.14, MAPE: 0.0861, RMSE: 6.35
Horizon 07, MAE: 3.25, MAPE: 0.0903, RMSE: 6.62
Horizon 08, MAE: 3.34, MAPE: 0.0941, RMSE: 6.86
Horizon 09, MAE: 3.43, MAPE: 0.0976, RMSE: 7.07
Horizon 10, MAE: 3.51, MAPE: 0.1009, RMSE: 7.26
Horizon 11, MAE: 3.59, MAPE: 0.1039, RMSE: 7.44
Horizon 12, MAE: 3.66, MAPE: 0.1069, RMSE: 7.61
```

Multi-step forecasting on 80 epochs:

```
--------multi-step testing results--------
Horizon 01, MAE: 2.21, MAPE: 0.0526, RMSE: 3.80
Horizon 02, MAE: 2.36, MAPE: 0.0575, RMSE: 4.24
Horizon 03, MAE: 2.48, MAPE: 0.0616, RMSE: 4.59
Horizon 04, MAE: 2.58, MAPE: 0.0652, RMSE: 4.88
Horizon 05, MAE: 2.67, MAPE: 0.0684, RMSE: 5.14
Horizon 06, MAE: 2.74, MAPE: 0.0714, RMSE: 5.36
Horizon 07, MAE: 2.82, MAPE: 0.0741, RMSE: 5.56
Horizon 08, MAE: 2.88, MAPE: 0.0766, RMSE: 5.74
Horizon 09, MAE: 2.94, MAPE: 0.0789, RMSE: 5.90
Horizon 10, MAE: 3.00, MAPE: 0.0811, RMSE: 6.05
Horizon 11, MAE: 3.05, MAPE: 0.0832, RMSE: 6.19
Horizon 12, MAE: 3.10, MAPE: 0.0852, RMSE: 6.32
```

