# Setup
Tested with `torch==1.11.0`, `torchvision==0.12.0`

```
apt-get install --no-install-recommends ffmpeg
```

```
conda create -n adacat python=3.7

pip install torch torchvision wandb numpy pandas h5py torch_ema==0.3 tqdm typed-argument-parser matplotlib ffmpeg scikit-video
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl

cd tto && pip install -e . 
```
