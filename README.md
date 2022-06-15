# Setup
Tested with `torch==1.11.0`, `torchvision==0.12.0`

```
apt-get install --no-install-recommends ffmpeg
```

```
conda create -n adacat python=3.7
conda activate adacat

pip install torch torchvision wandb numpy pandas h5py torch_ema==0.3 tqdm typed-argument-parser matplotlib ffmpeg scikit-video
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl

cd tto && pip install -e . 
```

# What is in this Repo
Code for reproducing the experiments in [https://openreview.net/forum?id=HMzzPOLs9l5](ADACAT: Adaptive Categorical Discretization for Autoregressive Models)

# Citation
The bibtex is provided below for citation covenience.
```
@inproceedings{
li2022adacat,
title={AdaCat: Adaptive Categorical Discretization for Autoregressive Models},
author={Qiyang Li and Ajay Jain and Pieter Abbeel},
booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
year={2022},
url={https://openreview.net/forum?id=HMzzPOLs9l5}
}
```

