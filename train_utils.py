import os
import json 
import yaml
import copy
import torch
import hydra
import random 
import argparse
import datetime
import pickle as pkl
import numpy as np
import torch.nn as nn
from os.path import join
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast, GradScaler
from einops.layers.torch import Rearrange, Reduce
from mosaic.utils.lr_scheduler import build_scheduler
from einops import rearrange, reduce, repeat, parse_shape
from mosaic.models.discrete_logistic import DiscreteMixLogistic
from collections import defaultdict, OrderedDict
from hydra.utils import instantiate
from mosaic.datasets.multi_task_datasets import BatchMultiTaskSampler, DIYBatchSampler, collate_by_task # need for val. loader
from torch.utils.data._utils.collate import default_collate
torch.autograd.set_detect_anomaly(True)
import learn2learn as l2l
# for visualization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,3,1,1))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,3,1,1))

def loss_to_scalar(loss):
    x = loss.item()
    return float("{:.5f}".format(x))

def make_data_loaders(config, dataset_cfg):
    """
    use yaml  to return train and val dataloaders
    """
    assert '_target_' in dataset_cfg.keys(), "Let's use hydra-config from now on. "
    print("Initializing {} with hydra config. \n".format(dataset_cfg._target_))
    #if dataset_cfg.get('agent_dir', None):
    #print("Agent file dirs: ", dataset_cfg.root_dir)
    dataset_cfg.mode = 'train'
    dataset = instantiate(dataset_cfg)
    samplerClass = DIYBatchSampler if config.samplers.use_diy else BatchMultiTaskSampler
    train_sampler = samplerClass(
            task_to_idx=dataset.task_to_idx,
            subtask_to_idx=dataset.subtask_to_idx,
            tasks_spec=dataset_cfg.tasks_spec,
            sampler_spec=config.samplers)
    #print("Dataloader has batch size {} \n".format(.batch_size))
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=min(11, config.get('loader_workers', cpu_count())),
        worker_init_fn=lambda w: np.random.seed(np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task
        )
         
    dataset_cfg.mode = 'val'
    val_dataset = instantiate(dataset_cfg)
    config.samplers.batch_size = config.train_cfg.val_size # allow validation batch to have a different size
    val_sampler = samplerClass(
            task_to_idx=val_dataset.task_to_idx,
            subtask_to_idx=val_dataset.subtask_to_idx,
            tasks_spec=dataset_cfg.tasks_spec,
            sampler_spec=config.samplers
            )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=min(11, config.get('loader_workers', cpu_count())),
        worker_init_fn=lambda w: np.random.seed(np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task
        )
    #print("Validation loader has {} total samples".format(len(val_loader)))
    return train_loader, val_loader

def collect_stats(step, task_losses, raw_stats, prefix='train'):
    """ create/append to stats dict of a one-layer dict structure:
        {'task_name/loss_key': [..], 'loss_key/task_name':[...]}"""
    task_names = sorted(task_losses.keys())
    for task, stats in task_losses.items():
        # expects: {'button': {"loss_sum": 1, "l_bc": 1}} 
        for k, v in stats.items(): 
            for log_key in [ f"{prefix}/{task}/{k}", f"{prefix}/{k}/{task}" ]:
                if log_key not in raw_stats.keys():
                    raw_stats[log_key] = []
                raw_stats[log_key].append(loss_to_scalar(v))
        if "step" in raw_stats.keys():
            raw_stats["step"].append(int(step)) 
        else:
            raw_stats["step"] = [int(step)]
    tr_print = ""
    for i, task in enumerate(task_names): 
        tr_print += "[{0:<9}] l_tot: {1:.1f} l_bc: {2:.1f} l_aux: {3:.1f} l_aux: {4:.1f} ".format( \
            task, 
            raw_stats[f"{prefix}/{task}/loss_sum"][-1], 
            raw_stats[f"{prefix}/{task}/l_bc"][-1],
            raw_stats.get(f"{prefix}/{task}/point_loss",[0])[-1], 
            raw_stats.get(f"{prefix}/{task}/l_aux",[0])[-1]) 
        if i % 3 == 2: # use two lines to print 
            tr_print += "\n"

    return tr_print

def generate_figure(images, context, fname='burner.png'):
    _B, T_im, _, _H, _W = images.shape
    T_con = context.shape[1]
    print("Images value range: ", images.min(), images.max(), context.max())
    print("Generating figures from images shape {}, context shape {} \n".format(images.shape, context.shape))
    npairs = 7
    skip   = 8
    ncols  = 4
    fig, axs = plt.subplots(nrows=npairs * 2, ncols=ncols, figsize=(ncols*3.5, npairs*2*2.8), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
    for img_index in range(npairs):
        show_img = images[img_index*skip].cpu().numpy() * STD + MEAN
        show_con = context[img_index*skip].cpu().numpy() * STD + MEAN
        for count in range(ncols):
            axs[img_index*2, count].imshow(show_img[count].transpose(1,2,0))
            if count < T_con:
                axs[img_index*2+1, count].imshow(show_con[count].transpose(1,2,0))

    plt.tight_layout()
    print("Saving figure to: ", fname)
    plt.savefig(fname)

def calculate_maml_loss(config, device, meta_model, model_inputs):
    states, actions = model_inputs['states'], model_inputs['actions']
    images, context = model_inputs['images'], model_inputs['demo']
    aux = model_inputs['aux_pose']

    meta_model = meta_model.to(device)
    inner_iters = config.daml.get('inner_iters', 1)
    l2error = torch.nn.MSELoss()

    #error = 0
    bc_loss, aux_loss = [], []

    for task in range(states.shape[0]):
        learner = meta_model.module.clone()
        for _ in range(inner_iters):
            learner.adapt(\
            learner(None, context[task], learned_loss=True)['learned_loss'], allow_nograd=True, allow_unused=True)
        out = learner(states[task], images[task], ret_dist=False)
        l_aux = l2error(out['aux'], aux[task][None])[None]
        mu, sigma_inv, alpha = out['action_dist']
        action_distribution = DiscreteMixLogistic(mu[:-1], sigma_inv[:-1], alpha[:-1])
        l_bc = -torch.mean(action_distribution.log_prob(actions[task]))[None]
        #validation_loss = l_bc + l_aux
        #error += validation_loss / states.shape[0]
        bc_loss.append(l_bc)
        aux_loss.append(l_aux)

    return torch.cat(bc_loss, dim=0), torch.cat(aux_loss, dim=0)

def calculate_task_loss(config, train_cfg, device, model, task_inputs):
    """Assumes inputs are collated by task names already, organize things properly before feeding into the model s.t.
        for each batch input, the model does only one forward pass."""
    all_loss, all_stats = dict(), dict()

    model_inputs = defaultdict(list)
    task_to_idx = dict()
    task_losses = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(task_inputs.items()):
        traj = inputs['traj']
        input_keys = ['states', 'actions', 'images', 'images_cp']
        if config.use_daml:
            input_keys.append('aux_pose')
        for key in input_keys:
            model_inputs[key].append( traj[key].to(device) )

        model_inputs['points'].append( traj['points'].to(device).long() )
        for key in ['demo', 'demo_cp']:
            model_inputs[key].append( inputs['demo_data'][key].to(device) )

        task_bsize  = traj['actions'].shape[0]
        task_to_idx[task_name] = [ start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        start += task_bsize

    for key in model_inputs.keys():
        model_inputs[key] = torch.cat(model_inputs[key], dim=0)
    all_losses = dict()
    if config.use_daml:
        bc_loss, aux_loss = calculate_maml_loss(model, model_inputs) 
        all_losses["l_bc"] = bc_loss
        all_losses["l_aux"] = aux_loss
        all_losses["loss_sum"] = bc_loss + aux_loss
    else:
        if config.policy._target_ == 'mosaic.models.mt_rep.VideoImitation':
            out = model(
                images=model_inputs['images'], images_cp=model_inputs['images_cp'], 
                context=model_inputs['demo'],  context_cp=model_inputs['demo_cp'],
                states=model_inputs['states'], ret_dist=False,
                multi_layer_actions=False,
                actions=model_inputs['actions']) 
        else: # other baselines 
            out = model(
                images=model_inputs['images'],
                context=model_inputs['demo'],
                states=model_inputs['states'], 
                ret_dist=False)
    
        # forward & backward action pred
        actions =  model_inputs['actions']
        mu_bc, scale_bc, logit_bc = out['bc_distrib'] # mu_bc.shape: B, 7, 8, 4]) but actions.shape: B, 6, 8
        action_distribution       = DiscreteMixLogistic(mu_bc[:,:-1], scale_bc[:,:-1], logit_bc[:,:-1])
        act_prob                  = rearrange(- action_distribution.log_prob(actions), 'B n_mix act_dim -> B (n_mix act_dim)')

        all_losses["l_bc"]     = train_cfg.bc_loss_mult * torch.mean(act_prob, dim=-1)
        # compute inverse model density
        inv_distribution       = DiscreteMixLogistic(*out['inverse_distrib'])
        inv_prob               = rearrange(- inv_distribution.log_prob(actions), 'B n_mix act_dim -> B (n_mix act_dim)')
        all_losses["l_inv"]    = train_cfg.inv_loss_mult * torch.mean(inv_prob, dim=-1)

        if 'point_ll' in out:
            pnts = model_inputs['points']
            l_point = train_cfg.pnt_loss_mult * \
                torch.mean(-out['point_ll'][range(pnts.shape[0]), pnts[:,-1,0], pnts[:,-1,1]], dim=-1)

            all_losses["point_loss"] = l_point

        # NOTE: the model should output calculated rep-learning loss
        rep_loss               = torch.zeros_like(all_losses["l_bc"] )
        for k, v in out.items():
            if k in train_cfg.rep_loss_muls.keys():
                v              = torch.mean(v, dim=-1) # just return size (B,) here
                v              = v * train_cfg.rep_loss_muls.get(k, 0)
                all_losses[k]  = v
                rep_loss       = rep_loss + v
        all_losses["rep_loss"] = rep_loss
        all_losses["loss_sum"] = all_losses["l_bc"] + all_losses["l_inv"] + rep_loss

    # flatten here to avoid headache
    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            if len(loss_val.shape) > 0:
                task_losses[task_name][loss_name] = torch.mean(loss_val[idxs])
    return task_losses