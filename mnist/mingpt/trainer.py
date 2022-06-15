"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, param, wandb_log=False):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.param = param
        self.wandb_log = wandb_log

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            print("CUDA is available :)")
            self.device = "cuda:0"
            self.model.cuda()
        else:
            print("CUDA is not available :(")

    def save_checkpoint(self, module=None):
        logger.info("saving %s", self.config.ckpt_path)
        if module is None:
            # DataParallel wrappers keep raw model object in .module attribute
            module = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(module.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer, ema = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch, curr_it):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            if not is_train:
                ema.store()
                ema.copy_to()
            losses = []
            # neg_ents = []
            total_norms = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    out_params = model(x)
                    
                    loss = self.param.compute_loss(out_params, y).mean()
                    losses.append(float(loss))

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()

                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip))
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    ema.update()

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

                    curr_it += 1

                    if self.wandb_log:
                        wandb.log({'grad_norm': grad_norm}, step=curr_it)
                        wandb.log({'train_loss_pb': float(loss)}, step=curr_it)
            if not is_train:
                ema.restore()
                info = {}
                test_eval_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_eval_loss)
                print()
                print("test loss: {:.3f}".format(test_eval_loss))
                info["curr_it"] = curr_it
                return test_eval_loss, info
            else:
                info = {}
                train_eval_loss = float(np.mean(losses))
                logger.info("train loss: %f", train_eval_loss)
                print()
                print("train loss: {:.3f}".format(train_eval_loss))
                info["curr_it"] = curr_it
                return train_eval_loss, info

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        yield 0, 0, best_loss

        curr_it = 0
        for epoch in range(config.max_epochs):

            train_eval_loss, info = run_epoch('train', epoch=epoch, curr_it=curr_it)
            curr_it = info['curr_it']
            if self.wandb_log:
                wandb.log({'train_eval_loss': train_eval_loss}, step=curr_it)
            if self.test_dataset is not None:
                test_eval_loss, _ = run_epoch('test', epoch=epoch, curr_it=curr_it)
                if self.wandb_log:
                    wandb.log({'test_eval_loss': test_eval_loss}, step=curr_it)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_eval_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_eval_loss
                self.save_checkpoint(ema)

            yield epoch + 1, curr_it, best_loss
