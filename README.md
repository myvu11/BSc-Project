# Bachelor thesis
Bachelor thesis about traffic flow prediciton.
In here the four models that have been worked with can be found. 

The data for the different models can be found here: [Models data](https://drive.google.com/drive/folders/1oM29OZrQfGAk-J2EvEO71PWYB1KT-OPw?usp=sharing)

The original data can be found here:
[Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).


## Requirements
Python 3.8.10 is used for the implementations.
Install following requirements to run the models.


### Python packages
```bash
pip install -r requirements.txt
```

```bash
pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

### CUDA Toolkit
If CUDA is wanted to be used to run the models then install following:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```