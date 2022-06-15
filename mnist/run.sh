# Script to reproduce the experiments in the paper
export seed=0
export n_epochs=1

# AdaCat
n_comps_lst=(256 128 108 90 76 64 32 16)
for n_comps in ${n_comps_lst[*]}; do
    python train.py --wandb_log --base_config base.yaml --param adacat --n_comps $n_comps --seed $seed --n_epochs $n_epochs
done

# DMoL
n_comps_lst=(171 72 60 43 22 11)
for n_comps in ${n_comps_lst[*]}; do 
    python train.py --wandb_log --base_config base.yaml --param dmol --n_comps $n_comps --seed $seed --n_epochs $n_epochs
done

# Uniform
n_comps_lst=(256 216 180 152 128 64 32)
for n_comps in ${n_comps_lst[*]}; do 
    python train.py --wandb_log --base_config base.yaml --param dmol --n_comps $n_comps --fixed_x 1 --seed $seed --n_epochs $n_epochs
done

# Quantile
n_comps_lst=(256 216 180 152 128 64 32)
for n_comps in ${n_comps_lst[*]}; do 
    python train.py --wandb_log --base_config base.yaml --param dmol --n_comps $n_comps --fixed_y 1 --seed $seed --n_epochs $n_epochs
done
