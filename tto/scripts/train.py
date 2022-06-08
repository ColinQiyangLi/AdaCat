import os
import numpy as np
import torch
import pdb

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.models.transformers import GPT

from trajectory.utils.parameterization import get_d_class

from torch_ema import ExponentialMovingAverage


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.offline'

#######################
######## setup ########
#######################

args = Parser().parse_args('train')

#######################
####### dataset #######
#######################

env = datasets.load_environment(args.dataset)

sequence_length = args.subsampled_sequence_length * args.step

if args.new:
    dataset_config = utils.Config(
        datasets.ContinuousDataset,
        savepath=(args.savepath, 'data_config.pkl'),
        env=args.dataset,
        penalty=args.termination_penalty,
        sequence_length=sequence_length,
        step=args.step,
        discount=args.discount,
    )
else:
    dataset_config = utils.Config(
        datasets.DiscretizedDataset,
        savepath=(args.savepath, 'data_config.pkl'),
        env=args.dataset,
        N=args.N,
        penalty=args.termination_penalty,
        sequence_length=sequence_length,
        step=args.step,
        discount=args.discount,
        discretizer=args.discretizer,
    )

dataset = dataset_config()
obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
transition_dim = dataset.joined_dim

#######################
######## model ########
#######################

block_size = args.subsampled_sequence_length * transition_dim - 1
print(
    f'Dataset size: {len(dataset)} | '
    f'Joined dim: {transition_dim} '
    f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
)


#######################
####### trainer #######
#######################

warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
final_tokens = 20 * warmup_tokens

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    # optimization parameters
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1, # only applied on matmul weights
    # learning rate decay: linear warmup followed by cosine decay to 10% of original
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
    ## dataloader
    num_workers=0,
    device=args.device,
)

trainer = trainer_config()

if args.eval_only == 1:
    gpt, gpt_epoch, gpt_ema = utils.load_model(args.logbase, args.dataset, args.exp_name,
            epoch='latest', device=args.device)
    gpt.register_d_class(d_class=get_d_class(args.d_class))

    if gpt_ema is not None:
        with gpt_ema.average_parameters():
            loss = trainer.evaluate(gpt, dataset)
    else:
        loss = trainer.evaluate(gpt, dataset)

    print("Eval Loss: {:.3f}".format(loss))

else:
    model_config = utils.Config(
        GPT,
        savepath=(args.savepath, 'model_config.pkl'),
        ## discretization
        vocab_size=args.N, block_size=block_size,
        ## architecture
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd*args.n_head,
        ## dimensions
        observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
        ## loss weighting
        action_weight=args.action_weight, reward_weight=args.reward_weight, value_weight=args.value_weight,
        ## dropout probabilities
        embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
        new=args.new,
        param_size=args.n_knobs * 2,
        smooth_coeff=args.smooth_coeff,
        smooth_type=args.smooth_type,
    )

    model = model_config()
    model.to(args.device)
    model.register_d_class(d_class=get_d_class(args.d_class))

    if hasattr(dataset, "discretizer"):
        model.register_discretizer(dataset.discretizer)

    if args.ema_decay != 0.:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        ema = None
        

    #######################
    ###### main loop ######
    #######################

    ## scale number of epochs to keep number of updates constant
    n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref)
    save_freq = int(n_epochs // args.n_saves)

    for epoch in range(n_epochs):
        print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} | {args.exp_name}')

        trainer.train(model, dataset, ema=ema)

        ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
        save_epoch = (epoch + 1) // save_freq * save_freq
        statepath = os.path.join(args.savepath, f'state_{save_epoch}.pt')
        print(f'Saving model to {statepath}')

        if ema is not None:
            model_state = model.state_dict()
            ema_state = ema.state_dict()
            state = {
                "model": model_state,
                "ema": ema_state,
                "ema_decay": args.ema_decay,
            }
            torch.save(state, statepath)
        else:
        ## save state to disk
            state = model.state_dict()
            torch.save(state, statepath)
