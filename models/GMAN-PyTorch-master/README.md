# PyTorch implementation of GMAN: A Graph Multi-Attention Network for Traffic Prediction

This is a testing PyTorch version implementation of Graph Multi-Attention Network in the following paper: Chuanpan Zheng, Xiaoliang Fan, Cheng Wang, and Jianzhong Qi. "[GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://arxiv.org/abs/1911.08415)", AAAI2020.



##  Requirements and dataset
The model is implemented using Python3.8.10 with dependencies specified in requirements.txt in main directory.

Download data from the directory [data_GMAN.zip](https://drive.google.com/drive/folders/1oM29OZrQfGAk-J2EvEO71PWYB1KT-OPw?usp=sharing)




## Training

Train on PEMS-BAY:
```bash
python train_on_data.py --model_file ./data/GMAN-pems.pkl --log_file log_GMAN_train_pems_ver --batch_size 32 --max_epoch 100 --patience 100
```

Train on METR-LA:
```bash
python train_on_data.py --traffic_file ./data/metr-la.h5 --SE_file ./data/SE\(METR\).txt --model_file ./data/GMAN-metr.pkl --log_file log_GMAN_train_metr_ver1 --batch_size 32 --max_epoch 100 --patience 100
```


## Testing

Test on PEMS-BAY:
```bash
python test_on_trained_data.py --model_file ./data/GMAN-pems.pkl --log_file log_GMAN_test_pems
```

Test on METR-LA:
```bash
python test_on_trained_data.py --traffic_file ./data/metr-la.h5 --SE_file ./data/SE\(METR\).txt --model_file ./data/GMAN-metr.pkl --log_file log_GMAN_test_metr
```


## Citation

This version of implementation is only for learning purpose. For research, please refer to  and  cite from the following paper:
```
@inproceedings{ GMAN-AAAI2020,
  author = "Chuanpan Zheng and Xiaoliang Fan and Cheng Wang and Jianzhong Qi"
  title = "GMAN: A Graph Multi-Attention Network for Traffic Prediction",
  booktitle = "AAAI",
  pages = "1234--1241",
  year = "2020"
}
```
