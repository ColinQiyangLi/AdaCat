# Miniboone
python train.py --n_knobs 300 --hidden_size 500 --n_layers 4 --dataset miniboone --fourier_k 8 --batch_size 1000
python train.py --n_knobs 200 --hidden_size 500 --n_layers 4 --dataset miniboone --fourier_k 8 --batch_size 1000
python train.py --n_knobs 100 --hidden_size 500 --n_layers 4 --dataset miniboone --fourier_k 8 --batch_size 1000

python train.py --n_knobs 300 --hidden_size 500 --n_layers 4 --dataset miniboone --fourier_k 8 --batch_size 1000 --fixed_x 1
python train.py --n_knobs 200 --hidden_size 500 --n_layers 4 --dataset miniboone --fourier_k 8 --batch_size 1000 --fixed_x 1
python train.py --n_knobs 100 --hidden_size 500 --n_layers 4 --dataset miniboone --fourier_k 8 --batch_size 1000 --fixed_x 1

# GAS
python train.py --n_knobs 1000 --hidden_size 1000 --n_layers 4 --dataset gas --fourier_k 32
python train.py --n_knobs 500  --hidden_size 1000 --n_layers 4 --dataset gas --fourier_k 32
python train.py --n_knobs 200  --hidden_size 1000 --n_layers 4 --dataset gas --fourier_k 32

python train.py --n_knobs 1000 --hidden_size 1000 --n_layers 4 --dataset gas --fourier_k 32 --fixed_x 1
python train.py --n_knobs 500  --hidden_size 1000 --n_layers 4 --dataset gas --fourier_k 32 --fixed_x 1
python train.py --n_knobs 200  --hidden_size 1000 --n_layers 4 --dataset gas --fourier_k 32 --fixed_x 1

# Hepmass
python train.py --n_knobs 300 --hidden_size 500 --n_layers 4 --dataset hepmass --fourier_k 4
python train.py --n_knobs 200 --hidden_size 500 --n_layers 4 --dataset hepmass --fourier_k 4
python train.py --n_knobs 100 --hidden_size 500 --n_layers 4 --dataset hepmass --fourier_k 4

python train.py --n_knobs 300 --hidden_size 500 --n_layers 4 --dataset hepmass --fourier_k 4 --fixed_x 1
python train.py --n_knobs 200 --hidden_size 500 --n_layers 4 --dataset hepmass --fourier_k 4 --fixed_x 1
python train.py --n_knobs 100 --hidden_size 500 --n_layers 4 --dataset hepmass --fourier_k 4 --fixed_x 1

# POWER
python train.py --n_knobs 500 --hidden_size 500 --n_layers 4 --dataset power --fourier_k 32 --smooth_coeff 0.00001
python train.py --n_knobs 300 --hidden_size 500 --n_layers 4 --dataset power --fourier_k 32 --smooth_coeff 0.00001
python train.py --n_knobs 100 --hidden_size 500 --n_layers 4 --dataset power --fourier_k 32 --smooth_coeff 0.00001 

python train.py --n_knobs 500 --hidden_size 500 --n_layers 4 --dataset power --fourier_k 32 --fixed_x 1 --smooth_coeff 0.00001
python train.py --n_knobs 300 --hidden_size 500 --n_layers 4 --dataset power --fourier_k 32 --fixed_x 1 --smooth_coeff 0.00001
python train.py --n_knobs 100 --hidden_size 500 --n_layers 4 --dataset power --fourier_k 32 --fixed_x 1 --smooth_coeff 0.00001
