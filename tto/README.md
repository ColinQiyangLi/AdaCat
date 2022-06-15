# Trajectory Transformer on D4RL datasets

## Training
```
python scripts/train.py --dataset halfcheetah-medium-v2 --ema_decay 0.995 --seed 0 --exp_name gpt/adacat_ema_0
python scripts/train.py --dataset walker2d-medium-v2    --ema_decay 0.995 --seed 0 --exp_name gpt/adacat_ema_0
python scripts/train.py --dataset hopper-medium-v2                        --seed 0 --exp_name gpt/adacat_0
```

We have provided [pretrained models](https://www.dropbox.com/s/1u6j0ybe0l4vh1l/adaca-d4rl-logs.zip?dl=0) for 3 datasets: `{halfcheetah, hopper, walker2d, ant}-{medium-v2}`. Download them with `./pretrained.sh`. You can directly plan with these pretrained models and skip the training step.

## Planning
```
./plan.sh
```

We also have provided [rollous](https://www.dropbox.com/s/6q5tsgl2kxriimh/adaca-d4rl-logs-wplan.zip?dl=0) for these pretrained models (3 rollouts for each model). Download them with `./pretrained_with_plan.sh` (will be stored in `logs_wplan`).

## Acknowledgements
The codebase is modified from Michael Janner's [trajectory_transformer](https://github.com/jannerm/trajectory-transformer) repo.
