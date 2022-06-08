# you're on your own to define a class that returns individual examples as PyTorch LongTensors
from torch.utils.data import Dataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample

import torch
import torchvision
import math
import random
import argparse

import os
from pathlib import Path
import sys
import random
import wandb
import numpy as np

import yaml

def manual_parse(lst, base_config, verbose=False):

    if verbose:
        print("{key:^20} | {value:^28}".format(key="Argument", value="Value"))
        print("-" * 50)

    new_config = base_config.copy()
    key = None
    for item in lst:
        if item.startswith("--"):
            assert key is None, "unrecognized string format: --{key} {item}".format(key=key, item=item)
            key = item[2:]
            assert key in base_config, "unrecognized key: {key}".format(key=key)
        else:
            assert key is not None, "unrecognized string format: {item} (expecting \'--\')".format(item=item)
            prev_value = base_config[key]
            typ = type(prev_value)
            new_value = typ(item)   # cast the item to the appropriate type
            new_config[key] = new_value
            key = None
    
    if verbose:
        for key, prev_value in base_config.items():
            new_value = new_config[key]
            if prev_value != new_value:
                print("{key:>20}: {prev_value:>12} => {new_value:<12}".format(
                    key=key, prev_value="" if prev_value == new_value else prev_value, new_value=new_value))
            else:
                print("{key:>20}: {prev_value:<16}{new_value:<12}".format(
                    key=key, prev_value="", new_value=new_value))
    
    return new_config


parser = argparse.ArgumentParser()
parser.add_argument('--base_config', type=str, default="base.yaml")
parser.add_argument('--project_name', type=str, default="adacat_mnist")
parser.add_argument('--entity', type=str, default=None, help="wandb entity (team or user), optional.")
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--wandb_log', action='store_true')
args, unknown = parser.parse_known_args()

with open(args.base_config, "r") as stream:
    base_config = yaml.safe_load(stream)

config = manual_parse(unknown, base_config, verbose=args.verbose)

if args.wandb_log:
    name = "-".join(["{key}:{value}".format(key=key[2:], value=value) for key, value in zip(unknown[::2], unknown[1::2])])
    print("Project Name:", args.project_name)
    print("Experiment Name:", name)
    wandb.init(project=args.project_name, name=name, config=config, entity=args.entity)

    config = wandb.config
else:
    # Aloow dot notation to access config properties
    class dotdict(dict):
        __getattr__ = dict.get

    config = dotdict(config)

np.random.seed(config.seed)
torch.random.manual_seed(config.seed)

class MNISTDataset(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        x = self.data[index].long()
        x = x.flatten()
        y = torch.nn.functional.pad(x[:-1], (1, 0))
        return y, x

class AdaCatParam:
    def __init__(self, n_comps):
        self.param_size = n_comps * 2

    def compute_loss(self, params, target):
        target = target[..., None]        
        target_left = target.float() / 256.
        target_right = target_left + 1. / 256.
        
        w_un, h_un = params[..., :config.n_comps], params[..., config.n_comps: config.n_comps * 2]
        if config.fixed_x:
            w_un = torch.zeros_like(w_un)
        if config.fixed_y:
            h_un = torch.zeros_like(h_un)

        x_sizes = torch.nn.functional.softmax(w_un, dim=-1)
        y_sizes = torch.nn.functional.softmax(h_un, dim=-1)
        
        y_sizes_cum = torch.cumsum(y_sizes, dim=-1)
        x_sizes_cum = torch.cumsum(x_sizes, dim=-1)
        
        indices_left = torch.searchsorted(x_sizes_cum, target_left).clamp(0, config.n_comps-1)
        indices_right = torch.searchsorted(x_sizes_cum, target_right).clamp(0, config.n_comps-1)
        
        y_sizes_cum = torch.nn.functional.pad(y_sizes_cum, (1, 0))
        x_sizes_cum = torch.nn.functional.pad(x_sizes_cum, (1, 0)) 

        y_left  = (target_left - x_sizes_cum.gather(-1, indices_left)) / x_sizes.gather(-1, indices_left) * y_sizes.gather(-1, indices_left) + y_sizes_cum.gather(-1, indices_left)
        y_right = (target_right - x_sizes_cum.gather(-1, indices_right)) / x_sizes.gather(-1, indices_right) * y_sizes.gather(-1, indices_right) + y_sizes_cum.gather(-1, indices_right)

        y_left = y_left.squeeze(-1)
        y_right = y_right.squeeze(-1)
        loss = -((y_right - y_left + 1e-6).log())
        bpd = loss / math.log(2.)
        return bpd

class DMoLParam:
    # original implementation from 
    def __init__(self, n_comps):
        self.param_size = n_comps * 3
    
    def compute_loss(self, params, target):
        """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
        l = params
        x = ((target.float() / 255. - 0.5) * 2.).clamp(min=-1., max=1.)
        xs = [int(y) for y in x.size()]
        ls = [int(y) for y in l.size()]

        # here and below: unpacking the params of the mixture of logistics
        nr_mix = ls[-1] // 3
        logit_probs = l[..., :nr_mix]
        l = l[..., nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
        means = l[..., :nr_mix]
        log_scales = torch.clamp(l[..., nr_mix:2 * nr_mix], min=-7.)
        # here and below: getting the means and adjusting them based on preceding
        # sub-pixels
        x = x.contiguous()
        x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix], device=x.device)

        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
        cdf_min = torch.sigmoid(min_in)
        # log probability for edge case of 0 (before scaling)
        log_cdf_plus = plus_in - torch.nn.functional.softplus(plus_in)
        # log probability for edge case of 255 (before scaling)
        log_one_minus_cdf_min = -torch.nn.functional.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases
        mid_in = inv_stdv * centered_x
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - log_scales - 2. * torch.nn.functional.softplus(mid_in)
        
        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
        inner_cond       = (x > 0.999).float()
        inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond             = (x < -0.999).float()
        log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
        log_probs        = log_probs + torch.nn.functional.log_softmax(logit_probs, dim=-1)

        return -torch.logsumexp(log_probs, dim=-1) / math.log(2)

train_dataset = MNISTDataset("data", train=True, download=True)
test_dataset = MNISTDataset("data", train=False, download=True)
img_size = (1, 28, 28)

inp_size = img_size[0] * img_size[1] * img_size[2]
batch_size = config.batch_size

if args.exp_name is None:
    import datetime
    exp_name = str(datetime.datetime.now())
else:
    exp_name = args.exp_name
exp_dir = os.path.join("exps", exp_name)
ckpt_path = os.path.join(exp_dir, "ckpt.pt")

Path(exp_dir).mkdir(parents=True, exist_ok=True)

if config.param == "dmol":
    param_cls = DMoLParam
elif config.param == "adacat":
    param_cls = AdaCatParam
else:
    raise ValueError("paramterization method {} not recognized".format(str(param_cls)))

param = param_cls(n_comps=config.n_comps)

print("# of output parameters: {}".format(param.param_size))

mconf = GPTConfig(
    vocab_size=256, 
    block_size=inp_size, 
    n_layer=config.n_layers, 
    n_head=config.n_head, 
    n_embd=config.n_embd, 
    param_size=param.param_size
) # a GPT-1
model = GPT(mconf)

# construct a trainer
tconf = TrainerConfig(
    max_epochs=config.n_epochs, batch_size=batch_size, n_gpus=config.n_gpus, ckpt_path=ckpt_path, 
    grad_norm_clip=config.grad_norm_clip, learning_rate=config.lr, weight_decay=config.weight_decay,
    ema_decay=config.get('ema_decay', 0)
)
trainer = Trainer(model, train_dataset, test_dataset, tconf, param=param, wandb_log=args.wandb_log)

for _ in trainer.train():
    pass
