from datasets import POWER, GAS, HEPMASS, MINIBOONE, BSDS300
from adacat import Adacat
import torch
import numpy as np
import yaml
import wandb
import os

from torch_ema import ExponentialMovingAverage

from tn import TruncatedNormal, TruncatedUniform
import random
import argparse


def batch_iter(X, batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])

    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


class MLP(torch.nn.Module):
    def __init__(self, *units, residual=False):
        super().__init__()

        self.layers = []
        for in_units, out_units in zip(units[:-1], units[1:]):
            self.layers.append(torch.nn.Linear(in_units, out_units))
        self.layers = torch.nn.ModuleList(self.layers)
        self.residual = residual

    def forward(self, x):
        y = x
        for index, layer in enumerate(self.layers):
            if index != 0: x = torch.nn.functional.softplus(x)
            x = layer(x)
        return x

class FF(torch.nn.Module):
    def __init__(self, n_dims, n_layers=3, hidden_size=128, out_size=256, out_d_class=Adacat, fixed_x=False, residual=False, fourier_k=0):
        super().__init__()

        self.init_d = torch.nn.Parameter(torch.randn(out_size))
        self.mlps = []
        for i in range(1, n_dims):
            self.mlps.append(MLP(*((i * (2 * fourier_k + 1),) + ((hidden_size,) * n_layers) + (out_size,)), residual=residual))
        self.mlps = torch.nn.ModuleList(self.mlps)
        self.out_d_class = out_d_class
        self.fixed_x = fixed_x
        self.fourier_k = fourier_k

    def forward(self, x):
        if self.fourier_k != 0:
            base = 2 ** torch.arange(self.fourier_k).to(device=device)
        ds = [self.init_d.expand(x.size(0), -1)]
        for i, mlp in zip(range(1, n_dims), self.mlps):
            inp = x[..., :i]
            if self.fourier_k != 0:
                inp_c = inp.unsqueeze(-1) * base
                inp_cos = torch.cos(inp_c)
                inp_sin = torch.sin(inp_c)
                inp_ff = torch.cat([inp_cos, inp_sin, inp.unsqueeze(-1)], dim=-1).view(-1, (2 * self.fourier_k + 1) * i)
                out = mlp(inp_ff)
            else:
                out = mlp(inp)
            ds.append(out)
        
        ds = torch.stack(ds, dim=-2)
        return ds

    def compute_loss(self, x):
        ds  = self.forward(x)
        if self.fixed_x:
            ds = torch.cat([ds, torch.zeros_like(ds)], dim=-1)
        ds  = self.out_d_class(ds)
        nll = -ds.log_prob(x)
        return nll.sum(dim=-1)

    def compute_smoothed_loss(self, x, coeff=0.0001):
        ds  = self.forward(x)
        if self.fixed_x:
            ds = torch.cat([ds, torch.zeros_like(ds)], dim=-1)
        ds  = self.out_d_class(ds)
        dx  = TruncatedNormal(loc=x, scale=torch.ones_like(x) * coeff, a=torch.zeros_like(x), b=torch.ones_like(x))
        nll = -ds.log_prob(dx)
        return nll.sum(dim=-1)

def load_data(name):

    if name == 'bsds300':
        return BSDS300()

    elif name == 'power':
        return POWER()

    elif name == 'gas':
        return GAS()

    elif name == 'hepmass':
        return HEPMASS()

    elif name == 'miniboone':
        return MINIBOONE()

    else:
        raise ValueError('Unknown dataset')

def save_checkpoint(model, optim, scheduler, ema, n_epoch, fname):
    state = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "n_epoch": n_epoch
    }
    torch.save(state, fname)

def load_checkpoint(model, optim, scheduler, ema, fname):
    state = torch.load(fname)
    model.load_state_dict(state["model"])
    optim.load_state_dict(state["optim"])
    scheduler.load_state_dict(state["scheduler"])
    ema.load_state_dict(state["ema"])
    return state["n_epoch"]


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
parser.add_argument('--project_name', type=str, default="adacat_tabular")
parser.add_argument('--verbose', action='store_true')
args, unknown = parser.parse_known_args()

with open(args.base_config, "r") as stream:
    base_config = yaml.safe_load(stream)

config = manual_parse(unknown, base_config, verbose=args.verbose)

base_dir = "logs"
exp_name = "{dataset}-{tag}-f{fourier_k}-n{n_layers}-h{hidden_size}-k{n_knobs}{residual}{fixed_x}-sm{smooth_coeff:.8f}-s{seed}".format(
    dataset=config["dataset"], tag=config["tag"], 
    fourier_k=config["fourier_k"],
    n_layers=config["n_layers"], 
    hidden_size=config["hidden_size"], n_knobs=config["n_knobs"], 
    residual="-residual" if config["residual"] else "",
    fixed_x="-fixed_x" if config["fixed_x"] else "",
    smooth_coeff=config["smooth_coeff"],
    seed=config["seed"])
exp_dir = os.path.join(base_dir, exp_name)

domain = ".".join(args.base_config.split("/")[1:])[:-5]
print("Project Name:", args.project_name)
print("Experiment Name:", exp_name)
wandb.init(project=args.project_name, name=exp_name, config=config)
config = wandb.config

seed = config.seed
dataset = config.dataset
batch_size = config.batch_size
test_batch_size = config.test_batch_size
n_knobs = config.n_knobs
n_epochs = config.n_epochs
hidden_size = config.hidden_size
n_layers = config.n_layers
tag = config.tag
device = "cuda:0"
fixed_x = config.fixed_x
residual = config.residual
fourier_k = config.fourier_k
smooth_coeff = config.smooth_coeff
weight_decay = config.weight_decay
lr = config.lr
ema_decay = config.ema_decay
scheduler_step_size = config.scheduler_step_size
scheduler_gamma = config.scheduler_gamma

load = False

os.makedirs(exp_dir, exist_ok = True)
def get_ckpt_dir(epoch=None):
    if epoch is None:
        return os.path.join(exp_dir, "ckpt.pt")
    return os.path.join(exp_dir, "ckpt_{}.pt".format(epoch))

result_dir = os.path.join(exp_dir, "result.json")
database_dir = os.path.join(base_dir, "final.json")


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data = load_data(dataset)

upper = np.maximum(np.amax(data.trn.x, axis=0), np.amax(data.tst.x, axis=0))
lower = np.minimum(np.amin(data.trn.x, axis=0), np.amin(data.tst.x, axis=0))

data.trn.x = torch.from_numpy(data.trn.x)
data.val.x = torch.from_numpy(data.val.x)
data.tst.x = torch.from_numpy(data.tst.x)

upper = torch.from_numpy(upper).to(device=device)
lower = torch.from_numpy(lower).to(device=device)
corr_const = float((upper - lower).log().sum())

n_dims = data.n_dims

model = FF(n_dims, hidden_size=hidden_size, n_layers=n_layers, out_size=n_knobs if fixed_x else n_knobs * 2, fixed_x=fixed_x, residual=residual, fourier_k=fourier_k)
model.to(device=device)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=scheduler_step_size, gamma=scheduler_gamma)

if load:
    ckpt_dir = get_ckpt_dir()
    n_epoch_st = load_checkpoint(model, optim, scheduler, ema, ckpt_dir)
    print("Checkpoint restored successfully. Starting at epoch {}".format(n_epoch_st))
else:
    n_epoch_st = 0

def actual(x):
    return x + corr_const

best_perf = None
for epoch in range(n_epoch_st, n_epochs):

    agg_train_losses, agg_train_eval_losses = [], []

    print("Epoch {}".format(epoch))
    for x in batch_iter(data.trn.x, batch_size=batch_size, shuffle=True):
        
        x = x.to(device=device)
        x = (x - lower) / (upper - lower)

        train_loss = model.compute_smoothed_loss(x, smooth_coeff)
        train_loss = train_loss.mean()

        optim.zero_grad()
        train_loss.backward()
        optim.step()
        
        ema.update()

        with torch.no_grad():
            train_eval_loss = model.compute_loss(x).mean()


        agg_train_losses.append(float(train_loss))
        agg_train_eval_losses.append(float(train_eval_loss))
        agg_train_mean = np.mean(agg_train_losses)
        agg_train_eval_mean = np.mean(agg_train_eval_losses)

        print("\rTrain Loss: {:.3f} [{:.3f}] (avg: {:.3f} [{:.3f}]) lr={}   ".format(actual(train_loss), actual(train_eval_loss), actual(agg_train_mean), actual(agg_train_eval_mean), scheduler.get_last_lr()), end="")
    
    scheduler.step()

    wandb.log({'train_loss': float(actual(agg_train_mean))}, step=epoch)
    wandb.log({'train_eval_loss': float(actual(agg_train_eval_mean))}, step=epoch)

    print()

    if epoch % 10 == 0 or epoch == n_epochs - 1:

        ema.store()
        ema.copy_to()
        agg_test_eval_losses = []
        for x in batch_iter(data.tst.x, batch_size=test_batch_size):
        
            x = x.to(device=device)
            x = (x - lower) / (upper - lower)

            with torch.no_grad():
                test_eval_loss = model.compute_loss(x).mean()
            agg_test_eval_losses.append(float(test_eval_loss))

        agg_test_eval_mean = np.mean(agg_test_eval_losses)
        if best_perf is None or agg_test_eval_mean < best_perf:
            best_perf = agg_test_eval_mean
        ema.restore()

        print("\rTest Loss at epoch {}: {:.3f}    ".format(epoch, actual(agg_test_eval_mean)))
        
        if epoch == n_epochs - 1:
            ckpt_dir = get_ckpt_dir(epoch=epoch if epoch != n_epochs else None)
            save_checkpoint(model, optim, scheduler, ema, epoch, ckpt_dir)
            print("Saved checkpoint at {}".format(ckpt_dir))
        
        wandb.log({'test_eval_loss': float(actual(agg_test_eval_mean))}, step=epoch)

    print()

final_test_nat = actual(best_perf)
result = {
    "dataset": dataset,
    "tag": tag,
    "seed": seed,
    "batch_size": batch_size,
    "test_batch_size": test_batch_size,
    "n_knobs": n_knobs,
    "n_epochs": n_epochs,
    "hidden_size": hidden_size,
    "n_layers": n_layers,
    "test_nat": final_test_nat,
} 

import json
with open(result_dir, "w") as write_file:
    json.dump(result, write_file, indent=4)

if os.path.exists(database_dir):
    with open(database_dir, "r") as f:
        data = json.load(f)
else:
    data = {}
data[exp_name] = final_test_nat

with open(database_dir, "w") as f:
    json.dump(data, f, indent=4)

print("database updated successfully at {}".format(database_dir))
