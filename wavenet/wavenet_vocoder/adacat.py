"""Adaptive Categorical Discretization"""

import torch
from torch.distributions import Distribution, constraints
import torch.nn.functional as F
from functools import reduce


def to_numpy(*args):
    return [arg.numpy() for arg in args]

def product(shape):
    if len(shape) == 0: return 1
    return reduce(lambda x, y: x * y, shape)

def gather_l(x, v):
    dim = len(x.shape)
    b_shape, f_shape = v.shape[:-dim+1], v.shape[-dim+1:]
    b, f = product(b_shape), product(f_shape)
    assert (x.shape[:-1] == f_shape)

    v = v.view(b, f).t()
    x = x.view(f, -1)

    return x.gather(-1, v).t().reshape(*b_shape, *f_shape)

def searchsorted_l(x, v):
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

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        
        indices = torch.multinomial(self.y_sizes.reshape(-1, self.n_knobs), sample_shape.numel(), True).t()
        
        indices = indices.view(sample_shape.numel(), *self.y_sizes.shape[:-1])
        
        x_right = gather_l(self.x_cum, indices)
        x_size = gather_l(self.x_sizes, indices)
        x = x_right - x_size * torch.rand_like(x_size)

        ret = x.reshape(self._extended_shape(sample_shape))

        return ret
        
    def _validate_distribution(self, d):
        assert self.support == d.support
    
    def log_prob(self, value):
        if isinstance(value, Distribution):
            # if passed in a distribution of value, we compute the expectation of log_prob
            # the distribution should have the same batch shape as the current distribution
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
