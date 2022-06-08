import torch
from torch.distributions import Distribution, constraints, AffineTransform, TransformedDistribution
import torch.nn.functional as F
import numpy as np
from functools import reduce

def get_d_class(name):
    if name == "adacat": 
        def adacat(params):
            return Adacat(logits=params)
        return adacat

def to_numpy(*args):
    return [arg.numpy() for arg in args]

def product(shape):
    if len(shape) == 0: return 1
    return reduce(lambda x, y: x * y, shape)

def gather_l(x, v):
    shape = v.shape
    dim = len(x.shape)
    b_shape, f_shape = v.shape[:-dim+1], v.shape[-dim+1:]
    b, f = product(b_shape), product(f_shape)
    assert (x.shape[:-1] == f_shape)

    v = v.view(b, f).t()
    x = x.view(f, -1)

    return x.gather(-1, v).t().reshape(*b_shape, *f_shape)

def searchsorted_l(x, v):
    shape = v.shape
    dim = len(x.shape)
    b_shape, f_shape = v.shape[:-dim+1], v.shape[-dim+1:]
    b, f = product(b_shape), product(f_shape)
    assert x.shape[:-1] == f_shape

    v = v.view(b, f).t()
    x = x.view(f, -1)
    return torch.searchsorted(x, v).t().reshape(*b_shape, *f_shape)

class Adacat(Distribution):
    arg_constraints = {'logits': constraints.real}
    support = constraints.unit_interval

    @property
    def mean(self):
        return ((self.x_cum - self.x_sizes / 2.) * self.y_sizes).sum(dim=-1)

    @property
    def x_sizes(self):
        return F.softmax(self.x_logits, dim=-1)

    @property
    def x_cum(self):
        return torch.cumsum(self.x_sizes, dim=-1)
    
    @property
    def y_sizes(self):
        return F.softmax(self.y_logits, dim=-1)
    
    @property
    def y_cum(self):
        return torch.cumsum(self.y_sizes, dim=-1)
    
    @property
    def log_x_sizes(self):
        return F.log_softmax(self.x_logits, dim=-1)
    
    @property
    def log_y_sizes(self):
        return F.log_softmax(self.y_logits, dim=-1)

    def __init__(self, logits, validate_args=None):
        self.logits = logits
        assert logits.size(-1) % 2 == 0
        self.n_knobs = logits.size(-1) // 2
        self.x_logits, self.y_logits = logits.split(self.n_knobs, dim=-1)

        batch_shape = self.x_logits.size()[:-1]
        super(Adacat, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Adacat, _instance)
        batch_shape = torch.Size(batch_shape)
        new.x_logits = self.x_logits.expand(batch_shape + (self.n_knobs,))
        new.y_logits = self.y_logits.expand(batch_shape + (self.n_knobs,))
        super(Adacat, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def sample(self, sample_shape=torch.Size(), zo_ratio=None, top_k=None, level=None, mean_sampling=None):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        # remove bins with low probabilities
        y_probs = self.y_sizes.reshape(-1, self.n_knobs)
        if zo_ratio is not None or top_k is not None: 
            assert not (zo_ratio is not None and top_k is not None) and level is None
            if zo_ratio is not None:
                nzo = int(self.n_knobs * zo_ratio)
            if top_k is not None:
                nzo = self.n_knobs - top_k
            _, indices = torch.topk(y_probs, k=nzo, dim=-1, largest=False)
            y_probs.scatter_(dim=-1, index=indices, src=torch.zeros_like(indices).float())
            
        if level is not None:
            y_sizes = self.y_sizes.reshape(-1, self.n_knobs)
            x_sizes = self.x_sizes.reshape(-1, self.n_knobs)

            density = y_sizes / x_sizes
            indices = density.argsort(dim=-1)
            y_sizes_sorted = y_sizes.gather(dim=-1, index=indices)
            
            y_sizes_sorted_prefix = torch.cumsum(y_sizes_sorted, dim=-1)

            sorted_cutoff = torch.searchsorted(y_sizes_sorted_prefix, level * torch.ones_like(y_sizes[:, 0:1]))

            reverse_indices = indices.argsort(dim=-1)
            mn_indices = torch.arange(0, self.n_knobs).to(x_sizes.device)

            # be a little bit conservative
            cutoff_mask = sorted_cutoff - 1 >= reverse_indices
            # zero out the entries that need to be cut off

            reverse_indices[cutoff_mask] = self.n_knobs
            y_sizes_sorted = torch.nn.functional.pad(y_sizes_sorted, (0, 1))
            y_probs = y_sizes_sorted.gather(dim=-1, index=reverse_indices)

        indices = torch.multinomial(y_probs, sample_shape.numel(), True).t()
        indices = indices.view(sample_shape.numel(), *self.y_sizes.shape[:-1])

        x_right = gather_l(self.x_cum, indices)
        x_size = gather_l(self.x_sizes, indices)
        if mean_sampling:
            x = x_right - x_size * 0.5
        else:
            x = x_right - x_size * torch.rand_like(x_size)

        ret = x.reshape(self._extended_shape(sample_shape))

        return ret

    def _validate_distribution(self, d):
        assert self.support == d.support
    
    def log_prob(self, value):
        if isinstance(value, Distribution):
            # if passed in a distribution of value, we compute the expectation of log_prob
            # the distribution should have the same batch shape as the current distribution

            # self._validate_distribution(value)
            x_cum = F.pad(self.x_cum, (1, 0))
            shape = x_cum.size()[:-1]
            numel = product(shape)
            x_cum = x_cum.view(-1, self.n_knobs+1).t().view(self.n_knobs+1, *shape)
            cdfs = value.cdf(x_cum.clamp(min=0., max=1.))
            ws = (cdfs[1:] - cdfs[:-1]).view(self.n_knobs, numel).t().view(*shape, self.n_knobs)
            return (ws * (self.log_y_sizes - self.log_x_sizes)).sum(dim=-1)
        
        else:
            # compute this normally

            if self._validate_args:
                self._validate_sample(value)

            indices = searchsorted_l(self.x_cum, value).clamp(
                0, self.n_knobs-1)

            log_x_size = gather_l(self.log_x_sizes, indices)
            log_y_size = gather_l(self.log_y_sizes, indices)

            return log_y_size - log_x_size

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        indices = searchsorted_l(self.x_cum, value).clamp(
            0, self.n_knobs-1)

        x_cum = gather_l(self.x_cum, indices)
        y_cum = gather_l(self.y_cum, indices)
        x_size = gather_l(self.x_sizes, indices)
        y_size = gather_l(self.y_sizes, indices)
        
        return y_cum - (x_cum - value) / x_size * y_size


    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        indices = searchsorted_l(self.y_cum, value).clamp(
            0, self.n_knobs-1)

        x_cum = gather_l(self.x_cum, indices)
        y_cum = gather_l(self.y_cum, indices)
        x_size = gather_l(self.x_sizes, indices)
        y_size = gather_l(self.y_sizes, indices)
        
        return x_cum - (y_cum - value) / y_size * x_size

    def entropy(self):
        return ((self.log_y_sizes - self.log_x_sizes) * self.x_sizes).sum(dim=-1)

if __name__ == "__main__":

    logits = torch.rand(10, 2, 20)
    d = Adacat(logits)
    print(d.sample().shape)

    # logits = torch.tensor([
    #     [1., 2., 3., 3., 2., 1.],
    #     [10., 10., 0., 0., 10., 0.],
    # ])

    # d = Adacat(logits)

    # import matplotlib.pyplot as plt

    # xs = torch.linspace(0.0, 1.0, 100).unsqueeze(-1).expand(-1, 2)
    # sps = d.sample((10000,))

    # probs = d.log_prob(xs).exp()

    # fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
    # ax1, ax2, ax3, ax4 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    # xs, probs, sps = to_numpy(xs, probs, sps)

    # ax1.plot(xs[:, 0], probs[:, 0], label="prob")
    # ax2.plot(xs[:, 1], probs[:, 1], label="prob")

    # ax3.clear()
    # ax3.hist(sps[:, 0], density=True)
    # ax4.hist(sps[:, 1], density=True)

    # def set_xlim(xlim, *args):
    #     for arg in args:
    #         arg.set_xlim(xlim)

    # set_xlim((0., 1.), ax1, ax2, ax3, ax4)

    # fig.savefig("adacat.png")

    # print(d.mean)
    # print(d.icdf(torch.tensor([0.6, 0.6])))
    # print(d.entropy())

    # from torch.distributions import ContinuousBernoulli
    # od = ContinuousBernoulli(torch.tensor([0.3, 0.8]))

    # print("Analytical:", d.log_prob(od))
    
    # ss = od.sample((100000,))

    # print(d.log_prob(ss).shape)

    # print("Empirical:", d.log_prob(ss).mean(dim=0))
