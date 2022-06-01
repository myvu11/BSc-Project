# DCRNN
This is a copy with some modification of the implementation that can be found here: https://github.com/KimMeen/DCRNN
It is a simplified PyTorch implementation of the paper https://arxiv.org/abs/1707.01926 based on [xlwang233/pytorch-DCRNN](https://github.com/xlwang233/pytorch-DCRNN) and [chnsh/DCRNN_PyTorch](https://github.com/chnsh/DCRNN_PyTorch)


The original tensorflow implementation: [liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)



## Requirements

The model is implemented using Python3.8.10 with dependencies specified in requirements.txt in bachelor directory

Download data from the directory [data_DCRNN.zip](https://drive.google.com/drive/folders/1oM29OZrQfGAk-J2EvEO71PWYB1KT-OPw?usp=sharing)

## Training

Firstly, you need to pre-process the data by using ```generate_dataset.py```:

# METR-LA
```bash
python -m generate_dataset --output_dir=data/processed/METR-LA --traffic_df_filename=data/metr-la.h5
```

# PEMS-BAY
```bash
python -m generate_dataset --output_dir=data/processed/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

Then execute the training script (default setup on METR-LA):

```bash
python train.py
```
 
 For training on PEMS-BAY 
```bash
python train.py --data data/PEMS-BAY --num_nodes 325 --checkpoints ./checkpoints/PEMS-BAY/dcrnn.pt --sensor_ids ./data/sensor_graph/graph_sensor_ids_bay.txt --sensor_distance ./data/sensor_graph/distances_bay_2017.csv --recording data/processed/PEMS-BAY --log_file log_DCRNN_train_bay.txt
```

## Testing

Run multi-step testing on trained model from horizon 1 to 12:

Test on METR-LA:
```bash
python multi_step_test.py --log_test log_DCRNN_test_la_ver.txt
```


Test on PEMS-BAY:
```bash
python multi_step_test.py --num_nodes 325 --checkpoints ./checkpoints/PEMS-BAY/dcrnn.pt --sensor_ids ./data/sensor_graph/graph_sensor_ids_bay.txt --sensor_distance ./data/sensor_graph/distances_bay_2017.csv --recording data/processed/PEMS-BAY --log_test log_DCRNN_test_bay.txt
```

