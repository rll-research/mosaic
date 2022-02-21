import torch
import torch.nn as nn
import numpy as np
from torch import einsum
import torch.nn.functional as F
from mosaic.models import get_model
 
import copy 
from einops import rearrange, reduce, repeat, parse_shape
from einops.layers.torch import Rearrange, Reduce
from itertools import chain
from collections import OrderedDict

def make_target(mlp):
    target = copy.deepcopy(mlp)
    target.load_state_dict(mlp.state_dict())
    for p in target.parameters():
        p.requires_grad = False 
    return target

class BYOLModule(nn.Module):
    """
    New(0509): add an option to share projector and predictor for pre&post attention features
            add another set to draw together mean-demonstration embeddings
    New(0506): assume the "target_out" embeddings come from the frozen target encoder + a stronger image augmentation
        Use this one module to try different video-representation learning loss: 
       First try BYOL: it's temporally-constrative + less concern about where to get negative samples
       Make this agnostic to teacher/student length: reshape before-hand, and
       just take in a batch of video frames, sample 'clips' (for now just length=1)
       If we want to do contrastive on _attended_ teacher/student videos, 
       either pass them in separate batches, or cut & concat with the same length
       default use only project dimension and hidden dimension for both projector and predictor
       """
    def __init__(
        self,
        embedder, 
        demo_T,
        obs_T,
        img_feat_dim, 
        attn_feat_dim,  
        img_conv_dim=256,
        attn_conv_dim=256,
        p=2,
        project_dim=128,
        hidden_dim=256,
        share_mlp=False,
        demo_proj=False,
        no_hidden=False,
        draw_apart=False,
        mul_pre=0,
        mul_pos=0,
        mul_demo=0,
        mul_intm=0,
        ):
        super().__init__()
        self.pre_attn_proj      = nn.Sequential(
            Rearrange('B T d H W -> (B T) d H W'),
            nn.BatchNorm2d(img_conv_dim), nn.ReLU(inplace=True),
            Rearrange('BT d H W -> BT (d H W)'),
            nn.Linear(img_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, project_dim)
            )
        if no_hidden:
            self.pre_attn_proj  = nn.Sequential(
                Rearrange('B T d H W -> (B T) d H W'),
                nn.BatchNorm2d(img_conv_dim), nn.ReLU(inplace=True),
                Rearrange('BT d H W -> BT (d H W)'),
                nn.Linear(img_feat_dim, project_dim),
                nn.LayerNorm(project_dim),
                )
        if share_mlp: # New(0518): use layernorm and hidden dim for BYOL 
            self.pre_attn_proj      = nn.Sequential(
                Rearrange('B T d H W -> (B T) d H W'),
                nn.BatchNorm2d(img_conv_dim), nn.ReLU(inplace=True),
                Rearrange('BT d H W -> BT (d H W)'), 
                nn.Linear(img_feat_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, project_dim),
                nn.LayerNorm(project_dim)
                )

        self.pre_attn_proj_tar  = make_target(self.pre_attn_proj)
        
        self.post_attn_proj     = nn.Sequential(
            # Rearrange('B T d H W -> (B T) d H W'),
            # nn.BatchNorm2d(attn_conv_dim), final attention layers already used batchnorm
            nn.ReLU(inplace=True), 
            Rearrange('B T d H W -> (B T) (d H W)'),
            nn.Linear(attn_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, project_dim))

        if no_hidden:
            self.post_attn_proj     = nn.Sequential(
                nn.ReLU(inplace=True), 
                Rearrange('B T d H W -> (B T) (d H W)'),
                nn.Linear(attn_feat_dim, project_dim),
                nn.LayerNorm(project_dim),
                )
         
        self.post_attn_proj_tar = make_target(self.post_attn_proj)
        # New(0509)
        self.share_mlp = share_mlp
        self.demo_proj = demo_proj
        if demo_proj:
            self.mean_demo_proj   = nn.Sequential(
                nn.BatchNorm2d(attn_conv_dim), nn.ReLU(inplace=True),
                Rearrange('B d H W -> B (d H W)'),
                nn.Linear(attn_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, project_dim)
                )
            self.mean_demo_proj_tar = make_target(self.mean_demo_proj)
            self.mean_demo_pred     = nn.Sequential(
                nn.Linear(project_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, project_dim))
        
        self.pre_attn_pred, self.post_attn_pred  = [
            nn.Sequential(
                nn.Linear(project_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, project_dim)) for _ in range(2) ] 
        if share_mlp:
            self.pre_attn_pred = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(project_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, project_dim), 
                nn.LayerNorm(project_dim))
            print("New(0519): Make BYOL MLPs more like SimCLR")

        self.draw_apart = draw_apart
        self.m_base             = 0.996 
        self.mom                = 0.996
        self.p                  = int(p)
        self._obs_T             = obs_T
        self._demo_T            = demo_T

    def forward(self, embed_out, embed_out_target):
        byol_out = OrderedDict()
        predict, project, target_project =  self.pre_attn_pred, self.pre_attn_proj, self.pre_attn_proj_tar
        feats, feats_aug = [out['img_features'] for out in [embed_out, embed_out_target]]
        byol_out['img_byol'] = \
            self.calculate_loss(feats, feats_aug, predict, project, target_project)
        
        if not self.share_mlp:
            predict, project, target_project =  self.post_attn_pred, self.post_attn_proj, self.post_attn_proj_tar
        feats, feats_aug = [out['attn_features'] for out in [embed_out, embed_out_target]]
        byol_out['attn_byol'] = \
            self.calculate_loss(feats, feats_aug, predict, project, target_project)
        
        # New(0512)
        feats, feats_aug = [out['attn_out_0'] for out in [embed_out, embed_out_target]]
        byol_out['intm_byol'] = \
            self.calculate_loss(feats, feats_aug, predict, project, target_project)
        
        if self.demo_proj:
            if not self.share_mlp:
                predict, project, target_project =  self.mean_demo_pred, self.mean_demo_proj, self.mean_demo_proj_tar
            feats, feats_aug = [out['demo_features'] for out in [embed_out, embed_out_target]] # note this is B, 1, d, H, W
            byol_out['demo_byol'] = \
                self.calculate_task_loss(feats, feats_aug, predict, project, target_project)
        return byol_out

    def calculate_loss(self, feats, feats_aug, predict, project, target_project):
        predicted             = predict(project(feats)) # -> B*T, proj_dim
        y                     = F.normalize(predicted, dim=-1, p=2)
        obs_T                 = feats.shape[1] - self._demo_T
        batch                 = feats.shape[0]
        proj_tar              = target_project(feats_aug)
        x                     = F.normalize(proj_tar, dim=-1, p=2)
        every_clip_loss       = - 2 * (x * y).sum(dim=-1)
        projected_target      = rearrange(proj_tar, '(B T) dim -> B T dim', B=batch)
        if self.p > 0:
            for _ in range(self.p):
                if self.draw_apart:
                    batch = projected_target.shape[0]
                    idxs  = torch.randperm(batch)
                    shuffled = rearrange(projected_target[idxs], 'B T dim ->(B T) dim')
                    x        = F.normalize(shuffled, dim=-1, p=2)
                    every_clip_loss   = every_clip_loss + 2 * (x * y).sum(dim=-1)
                else:
                    # the trick here is how to get the multiple 'key' clips for every clip in every video of the batch
                    # we do this sequentially: if we permute p times it's equivalent to caculating the summed loss over p augmented clips
                    tar_demo, tar_obs = projected_target.split([self._demo_T, obs_T], dim=1)
                    demo_idxs         = torch.randperm(self._demo_T)
                    obs_idxs          = torch.randperm(obs_T)
                    shuffled          = rearrange(
                        torch.cat([tar_demo[:, demo_idxs], tar_obs[:, obs_idxs]], dim=1), 'B T dim -> (B T) dim')
                    x                 = F.normalize(shuffled, dim=-1, p=2)
                    every_clip_loss   = every_clip_loss - 2 * (x * y).sum(dim=-1)
        return every_clip_loss / (self.p + 1) 

    def calculate_task_loss(self, demo, demo_aug, predict, project, target_project):
        demo_mean, demo_mean_aug = [ torch.mean(dem, dim=1, keepdim=self.share_mlp) for dem in [demo, demo_aug] ]
        # this is different shape: B, d, H, W
        y        = F.normalize(predict(project(demo_mean)), dim=-1, p=2)
        proj_tar = target_project(demo_mean_aug) # B, d
        x        = F.normalize(proj_tar, dim=-1, p=2)
        every_demo_loss       = - 2 * (x * y).sum(dim=-1)
        return every_demo_loss

    def update_mom(self, frac):
        """frac should be k/K, i.e. proportional to total training iterations"""
        assert 0 <= frac <= 1, 'we only decay momentum by k/K' 
        self.mom =  1 - (1 - self.m_base) * ( np.cos(np.pi*frac) + 1 ) / 2
    
    def soft_param_update(self):
        tau = 1 - self.mom 
        chained = chain(
            self.pre_attn_proj.parameters(),
            self.post_attn_proj.parameters(), 
            self.mean_demo_proj.parameters(),
        )
        chained_target = chain(
            self.pre_attn_proj_tar.parameters(),
            self.post_attn_proj_tar.parameters(), 
            self.mean_demo_proj_tar.parameters(),
        )
        for param, target_param in zip( chained, chained_target ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        return 
        

class ContrastiveModule(nn.Module):
    """
    NOTE: do compare instance-level constrastive loss, set k=0
    Random shuffle and contrast against other temporally apart frames within the batch.
    Use one set of MLPs as projector and predictor for any convolution features,
    and keep W matrices separate across input features from different attention layers
    """
    def __init__(
        self,
        embedder,
        demo_T,
        obs_T,
        img_conv_dim,
        attn_conv_dim,
        img_feat_dim,
        attn_feat_dim,
        tau=0.01,
        compressor_dim=128,
        temporal=False,
        hidden_dim=0,
        mul_pre=0,
        mul_pos=0,
        mul_intm=0,
        fix_step=-1,
        ):
        super().__init__()

        self.frame_compressor = nn.Sequential(
            Rearrange('B T d H W -> (B T) d H W'),
            nn.BatchNorm2d(img_conv_dim), nn.ReLU(inplace=True),
            Rearrange('BT d H W -> BT (d H W)'),
            nn.Linear(img_feat_dim, compressor_dim),
            nn.LayerNorm(compressor_dim)
            )
        self.predictor = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(compressor_dim, compressor_dim),
            nn.LayerNorm(compressor_dim)
            )
        if hidden_dim:
            print("Contrastive MLP using hidden dim: ", hidden_dim)
            self.frame_compressor = nn.Sequential(
                Rearrange('B T d H W -> (B T) d H W'),
                nn.BatchNorm2d(img_conv_dim), nn.ReLU(inplace=True),
                Rearrange('BT d H W -> BT (d H W)'),
                nn.Linear(img_feat_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, compressor_dim),
                nn.LayerNorm(compressor_dim)
                )
            self.predictor = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(compressor_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, compressor_dim),
                nn.LayerNorm(compressor_dim)
                )

        self.frame_compressor_target = make_target(self.frame_compressor)
        d_f                =   compressor_dim
        self._pre_attn_W   =   nn.Parameter(torch.rand(d_f, d_f))
        self._post_attn_W  =   nn.Parameter(torch.rand(d_f, d_f))
        self._intm_attn_W  =   nn.Parameter(torch.rand(d_f, d_f)) # New(0512)
        self.tau           =   tau
        self._demo_T       =   demo_T
        self.temporal      =   temporal
        self.fix_step      = fix_step

    def forward(self, embed_out, embed_out_target):
        sim_out = OrderedDict()
        self.calculate_loss(embed_out, embed_out_target, sim_out, in_key='img_features', out_key='simclr_pre')
        self.calculate_loss(embed_out, embed_out_target, sim_out, in_key='attn_features', out_key='simclr_post')
        self.calculate_loss(embed_out, embed_out_target, sim_out, in_key='attn_out_0', out_key='simclr_intm')
        return sim_out

    def calculate_loss(self, embed_out, embed_out_target, sim_out=None, in_key='img_features', out_key='pre'):
        img_anc     = embed_out.get(in_key, None)
        assert img_anc is not None, 'output is missing: '+str(in_key)
        compressor  = self.frame_compressor
        z_a         = compressor(img_anc) # BT, D
        if self.temporal:
            z_a     = self.predictor(z_a)
        img_pos     = embed_out_target[in_key]
        assert img_pos.shape == img_anc.shape
        if self.temporal:
            obs_T             = img_pos.shape[1] - self._demo_T
            tar_demo, tar_obs = img_pos.split([self._demo_T, obs_T], dim=1)
            if self.fix_step > 0: # eq to ATC
                k, dT = self.fix_step, self._demo_T
                assert k <= dT
                demo_idxs         = list(range(dT))[k:] + list(range(k))
                obs_idxs          = list(range(obs_T))[k:] + list(range(k))
            else:
                demo_idxs         = torch.randperm(self._demo_T)
                obs_idxs          = torch.randperm(obs_T)
            img_pos           = torch.cat([tar_demo[:, demo_idxs], tar_obs[:, obs_idxs]], dim=1)

        compressor_tar = self.frame_compressor_target
        z_pos = compressor_tar(img_pos).detach()

        # compute (B*T,B*T) matrix z_a (W z_pos.T)
        # use multiclass cross entropy with identity matrix for labels
        W_frame = self._pre_attn_W
        if 'pos' in out_key:
            W_frame = self._post_attn_W
        elif 'intm' in out_key:
            W_frame = self._intm_attn_W
        logits = einsum('ad,dd,bd->ab', z_a, W_frame, z_pos)
       
        logits = logits - reduce(logits, 'a b -> a 1', 'max')
        labels = torch.arange(logits.shape[0]).long().to(logits.get_device())

        sim_out[out_key] = F.cross_entropy(logits, labels, reduction='none')

        return

    def soft_param_update(self):
        tau = self.tau
        for param, target_param in zip( self.frame_compressor.parameters(), self.frame_compressor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        return
