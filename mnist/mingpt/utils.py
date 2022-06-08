import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample_fn=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    # block_size = model.get_block_size()
    model.eval()
    print()
    for k in range(steps):
        x_cond = x # if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        out_params = model(x_cond)
        s = sample_fn(out_params, temperature=temperature)
        x = torch.cat((x, s[:, -1:]), dim=1)
        print(" Sampling - [{}/{}]\r".format(k+1, steps), end="")
    print("Done.\n")
    return x
