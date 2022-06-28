# MNIST Density Modeling with AdaCat

This folder contains the code to reproduce the experiments in Figure 4 and Table 2 of the paper.

To reproduce the results in the paper, run
```
./run.sh
```

wandb logging is enabled by default. It can be turned off by removing the `--wandb_log` flag from the `python train.py` command.

## Command line arguments
- `--fixed_x 1`: if enabled, the width of each bin in the AdaCat parameterization remains fixed and is uniformly distributed between 0 and 1.
- `--fixed y 1`: if enabled, each bin always gets assigned the same probability. This is our Quantile baseline in Table 2. 
- `--n_comps [n_comps]`: controls the number of mixture components the distribution has (e.g., number of logistic distributions in discrete mixture of logistics or number of bins in AdaCat). 

## Acknowledgements
The code is built on top of the Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT)

