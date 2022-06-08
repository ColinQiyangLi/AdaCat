# Script to reproduce the experiments in the paper
seed = 0

# AdaCat
n_comps_lst=(256 128 108 90 76 64 32 16)
for n_comps in ${n_comps_lst[*]}; do 
    python train.py --base_config base.yaml --param adacat --n_comps n_comps --wandb_log --seed $seed

# DMoL
n_comps_lst=(171 72 60 43 22 11)
for n_comps in ${n_comps_lst[*]}; do 
    python train.py --base_config base.yaml --param dmol --n_comps n_comps --wandb_log --seed $seed

# Uniform
n_comps_lst=(256 216 180 152 128 64 32)
for n_comps in ${n_comps_lst[*]}; do 
    python train.py --base_config base.yaml --param dmol --n_comps n_comps --wandb_log --fixed 1 --seed $seed

# Quantile
n_comps_lst=(256 216 180 152 128 64 32)
for n_comps in ${n_comps_lst[*]}; do 
    python train.py --base_config base.yaml --param dmol --n_comps n_comps --wandb_log --fixed 1 --seed $seed
