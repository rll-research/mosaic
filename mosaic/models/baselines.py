"""
Contains baseline models: 
1) T-OSIL (transformer-based) from Sundeep et al.
2) LSTM architecture
3) Plain MLP model
4) MAML meta-learning model from Finn et al.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mosaic.models import get_model
from mosaic.models.basic_embeddings import NonLocalLayer, TemporalPositionalEncoding, TempConvLayer
from torchvision import models
from mosaic.models.discrete_logistic import DiscreteMixLogistic
import numpy as np
from torch.distributions import MultivariateNormal

class _TransformerFeatures(nn.Module):
    def __init__(self, latent_dim, conv_drop_dim=2, context_T=3, embed_hidden=256, dropout=0.2, n_st_attn=0, use_ss=True, st_goal_attn=False, use_pe=False, attn_heads=1, attn_ff=128, just_conv=False):
        super().__init__()
        self._resnet18 = get_model('resnet')(output_raw=True, drop_dim=conv_drop_dim, use_resnet18=True, pretrained=True)
        conv_out_dim = 256 if conv_drop_dim == 3 else 512
        ProcLayer = TempConvLayer if just_conv else NonLocalLayer
        self._temporal_process = nn.Sequential(ProcLayer(conv_out_dim, conv_out_dim, attn_ff, dropout=dropout, n_heads=attn_heads), nn.Conv3d(conv_out_dim, conv_out_dim, (context_T, 1, 1), 1))
        in_dim, self._use_ss = conv_out_dim * 2 if use_ss else conv_out_dim, use_ss
        self._to_embed = nn.Sequential(nn.Linear(in_dim, embed_hidden), nn.Dropout(dropout), nn.ReLU(), nn.Linear(embed_hidden, latent_dim))
        self._st_goal_attn = st_goal_attn
        self._st_attn = nn.Sequential(*[ProcLayer(conv_out_dim, conv_out_dim, 128, dropout=dropout, causal=True, n_heads=attn_heads) for _ in range(n_st_attn)])
        self._pe = TemporalPositionalEncoding(conv_out_dim, dropout) if use_pe else None

    def forward(self, images, context, forward_predict):
        assert len(images.shape) == 5, "expects [B, T, C, H, W] tensor!"
        im_in = torch.cat((context, images), 1) if forward_predict or self._st_goal_attn else images
        features = self._st_attn(self._resnet_features(im_in).transpose(1, 2)).transpose(1, 2)
        if forward_predict:
            features = self._temporal_process(features.transpose(1, 2)).transpose(1, 2)
            features = torch.mean(features, 1, keepdim=True)
        elif self._st_goal_attn:
            T_ctxt = context.shape[1]
            features = features[:,T_ctxt:]
        return F.normalize(self._to_embed(self._spatial_embed(features)), dim=2)

    def _spatial_embed(self, features):
        if not self._use_ss:
            return torch.mean(features, (3, 4))

        features = F.softmax(features.reshape((features.shape[0], features.shape[1], features.shape[2], -1)), dim=3).reshape(features.shape)
        h = torch.sum(torch.linspace(-1, 1, features.shape[3]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 4), 3)
        w = torch.sum(torch.linspace(-1, 1, features.shape[4]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 3), 3)
        return torch.cat((h, w), 2)

    def _resnet_features(self, x):
        if self._pe is None:
            return self._resnet18(x)
        features = self._resnet18(x).transpose(1, 2)
        features = self._pe(features).transpose(1, 2)
        return features

class _AblatedFeatures(_TransformerFeatures):
    def __init__(self, latent_dim, model_type='basic', temp_convs=False, lstm=False, context_T=2):
        nn.Module.__init__(self)

        # initialize visual network
        assert model_type in ('basic', 'resnet'), "Unsupported model!"
        self._visual_model = get_model('resnet')(output_raw=True, drop_dim=2, use_resnet18=True) if model_type == 'resnet' else get_model('basic')()
        ss_dim = 1024 if model_type == 'resnet' else 64
        self._use_ss, self._latent_dim = True, latent_dim

        # seperate module to process context if needed
        self._temp_convs = temp_convs
        if temp_convs:
            tc_dim = int(ss_dim / 2)
            fc1, a1 = nn.Linear(ss_dim, ss_dim), nn.ReLU(inplace=True)
            fc2, a2 = nn.Linear(ss_dim, ss_dim), nn.ReLU(inplace=True)
            self._fcs = nn.Sequential(fc1, a1, fc2, a2)
            self._tc = nn.Conv1d(ss_dim, tc_dim, context_T, stride=1)
        else:
            tc_dim = 0

        # go from input features to latent vector
        self._to_goal = nn.Sequential(nn.Linear((context_T + 1) * (tc_dim + ss_dim), tc_dim + ss_dim), nn.ReLU(inplace=True), nn.Linear(tc_dim + ss_dim, latent_dim))
        self._to_latent = nn.Sequential(nn.Linear(tc_dim + ss_dim, latent_dim), nn.ReLU(inplace=True), nn.Linear(latent_dim, latent_dim))

        # configure lstm network for sequential processing
        self._has_lstm = lstm
        if lstm:
            self._lstm_module = nn.LSTM(latent_dim, latent_dim, 1)

    def forward(self, images, context, forward_predict):
        feats = self._visual_model(torch.cat((context, images), 1))
        feats = self._spatial_embed(feats)

        if self._temp_convs:
            ctxt_feats = feats[:,:context.shape[1]]
            ctxt_feats = self._tc(self._fcs(ctxt_feats).transpose(1, 2)).transpose(1,2)
            feats = torch.cat((feats, ctxt_feats.repeat((1, feats.shape[1], 1))), 2)

        if forward_predict:
            goal_feats = feats[:,:context.shape[1] + 1].reshape((feats.shape[0], -1))
            return self._to_goal(goal_feats)[:,None]

        latents = self._to_latent(feats)
        latents = self._lstm(latents) if self._has_lstm else latents
        return latents[:,context.shape[1]:]

    def _lstm(self, latents):
        assert self._has_lstm, "needs lstm to forward!"
        self._lstm_module.flatten_parameters()
        return self._lstm_module(latents)[0]

class _DiscreteLogHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures, const_var=False):
        super().__init__()
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._n_mixtures, self._dist_size = n_mixtures, torch.Size((out_dim, n_mixtures))
        self._mu = nn.Linear(in_dim, out_dim * n_mixtures)
        if const_var:
            ln_scale = torch.randn(out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        else:
            self._ln_scale = nn.Linear(in_dim, out_dim * n_mixtures)
        self._logit_prob = nn.Linear(in_dim, out_dim * n_mixtures) if n_mixtures > 1 else None

    def forward(self, x):
        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))
        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(x).reshape((x.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)

        logit_prob = self._logit_prob(x).reshape((x.shape[:-1] + self._dist_size)) if self._n_mixtures > 1 else torch.ones_like(mu)
        return (mu, ln_scale, logit_prob)

class InverseImitation(nn.Module):
    def __init__(self, latent_dim, lstm_config, sdim=9, adim=8, n_mixtures=3, concat_state=True, const_var=False, pred_point=False, vis=dict(), transformer_feat=True):
        super().__init__()
        # initialize visual embeddings
        self._embed = _TransformerFeatures(latent_dim, **vis) if transformer_feat else _AblatedFeatures(latent_dim, **vis)

        # inverse modeling
        inv_dim = latent_dim * 2
        self._inv_model = nn.Sequential(nn.Linear(inv_dim, lstm_config['out_dim']), nn.ReLU())

        # additional point loss on goal
        self._2_point = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 5)) if pred_point else None

        # action processing
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._concat_state, self._n_mixtures = concat_state, n_mixtures
        self._is_rnn = lstm_config.get('is_rnn', True)
        if self._is_rnn:
            self._action_module = nn.LSTM(int(latent_dim + latent_dim + float(concat_state) * sdim), lstm_config['out_dim'], lstm_config['n_layers'])
        else:
            l1, l2 = [nn.Linear(int(latent_dim + latent_dim + float(concat_state) * sdim), lstm_config['out_dim']), nn.ReLU()], []
            for _ in range(lstm_config['n_layers'] - 1):
                l2.extend([nn.Linear(lstm_config['out_dim'], lstm_config['out_dim']), nn.ReLU()])
            self._action_module = nn.Sequential(*(l1 + l2))
        self._action_dist = _DiscreteLogHead(lstm_config['out_dim'], adim, n_mixtures, const_var)

    def forward(self, states, images, context, ret_dist=True, eval=False):
        img_embed = self._embed(images, context, False)
        pred_latent, goal_embed = self._pred_goal(images[:,:1], context)
        states = torch.cat((img_embed, states), 2) if self._concat_state else img_embed

        # run inverse model
        inv_in = torch.cat((img_embed[:,:-1], img_embed[:,1:]), 2)
        mu_inv, scale_inv, logit_inv = self._action_dist(self._inv_model(inv_in))

        # predict behavior cloning distribution
        ac_in = goal_embed.transpose(0, 1).repeat((states.shape[1], 1, 1))
        ac_in = torch.cat((ac_in, states.transpose(0, 1)), 2)
        if self._is_rnn:
            self._action_module.flatten_parameters()
            ac_pred = self._action_module(ac_in)[0].transpose(0, 1)
        else:
            ac_pred = self._action_module(ac_in.transpose(0, 1))
        mu_bc, scale_bc, logit_bc = self._action_dist(ac_pred)

        out = {}
        # package distribution in objects or as tensors
        if ret_dist:
            out['bc_distrib'] = DiscreteMixLogistic(mu_bc, scale_bc, logit_bc)
            out['inverse_distrib'] = DiscreteMixLogistic(mu_inv, scale_inv, logit_inv)
        else:
            out['bc_distrib'] = (mu_bc, scale_bc, logit_bc)
            out['inverse_distrib'] = (mu_inv, scale_inv, logit_inv)
        if eval:
            return out
        out['pred_goal'] = pred_latent
        out['img_embed'] = img_embed
        self._pred_point(out, goal_embed, images.shape[3:])
        return out

    def _pred_goal(self, img0, context):
        g_embed = self._embed(img0, context, True)
        return g_embed, g_embed

    def _pred_point(self, obs, goal_embed, im_shape, min_std=0.03):
        if self._2_point is None:
            return

        point_dist = self._2_point(goal_embed[:,0])
        mu = point_dist[:,:2]
        c1, c2, c3 = F.softplus(point_dist[:,2])[:,None], point_dist[:,3][:,None], F.softplus(point_dist[:,4])[:,None]
        scale_tril = torch.cat((c1 + min_std, torch.zeros_like(c2), c2, c3 + min_std), dim=1).reshape((-1, 2, 2))
        mu, scale_tril = [x.unsqueeze(1).unsqueeze(1) for x in (mu, scale_tril)]
        point_dist = MultivariateNormal(mu, scale_tril=scale_tril)

        h = torch.linspace(-1, 1, im_shape[0]).reshape((1, -1, 1, 1)).repeat((1, 1, im_shape[1], 1))
        w = torch.linspace(-1, 1, im_shape[1]).reshape((1, 1, -1, 1)).repeat((1, im_shape[0], 1, 1))
        hw = torch.cat((h, w), 3).repeat((goal_embed.shape[0], 1, 1, 1)).to(goal_embed.device)
        obs['point_ll'] = point_dist.log_prob(hw)


class DAMLNetwork(nn.Module):
    def __init__(self, n_final_convs, aux_dim=6, adim=8, n_mix=2, T_context=2, const_var=True, maml_lr=None, first_order=None):
        super().__init__()
        if n_final_convs == 'resnet':
            self._is_resnet = True
            self._visual_net = get_model('resnet')(output_raw=True, drop_dim=3, use_resnet18=True)
            self._resnet_2_embed = nn.Sequential(nn.Linear(512, 256), nn.Dropout(p=0.2), nn.ReLU(), nn.Linear(256, 256))
            n_final_convs = 128
        else:
            self._is_resnet = False
            vgg_pre = models.vgg16(pretrained=True).features[:5]
            c3, a3 = nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU()
            c4, a4 = nn.Conv2d(64, n_final_convs, 3, stride=2, padding=1), nn.ReLU()
            c5, a5 = nn.Conv2d(n_final_convs, n_final_convs, 3, stride=1, padding=1), nn.ReLU()
            self._visual_net = nn.Sequential(vgg_pre, c3, a3, c4, a4, c5, a5)
        self._aux_pose = nn.Sequential(nn.Linear(n_final_convs * 2, 32), nn.ReLU(), nn.Linear(32, aux_dim))
        self._action_mlp = nn.Sequential(nn.Linear(n_final_convs * 2, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU())
        self._action_dist = _DiscreteLogHead(50, adim, n_mix, const_var)
        dim1, dim2 = T_context * (50 + 2 * n_final_convs), 50 * T_context
        self._learned_loss = nn.Sequential(nn.Linear(dim1, dim2), nn.ReLU(), nn.LayerNorm(dim2), nn.Linear(dim2, dim2), nn.ReLU(), nn.LayerNorm(dim2), nn.Linear(dim2, 1))

    def forward(self, states, images, ret_dist=True, learned_loss=False):
        img_embed = self._embed_images(images)
        action_mlp = self._action_mlp(img_embed)

        out = {}
        if learned_loss:
            learned_in = torch.cat((action_mlp, img_embed), -1)
            out['learned_loss'] = torch.squeeze(self._learned_loss(learned_in.reshape((1, -1))))
        else:
            mu, sigma_inv, alpha = [x[0] for x in self._action_dist(action_mlp[None])]
            out['action_dist'] = DiscreteMixLogistic(mu, sigma_inv, alpha) if ret_dist else (mu, sigma_inv, alpha)
            out['aux'] = self._aux_pose(img_embed[:1])
        return out

    def _embed_images(self, images):
        features = self._visual_net(images)
        features = F.softmax(features.reshape((features.shape[0], features.shape[1], -1)), dim=2).reshape(features.shape)
        h = torch.sum(torch.linspace(-1, 1, features.shape[2]).view((1, 1, -1)).to(features.device) * torch.sum(features, 3), 2)
        w = torch.sum(torch.linspace(-1, 1, features.shape[3]).view((1, 1, -1)).to(features.device) * torch.sum(features, 2), 2)
        spatial_softmax = torch.cat((h, w), 1)
        if self._is_resnet:
            return self._resnet_2_embed(spatial_softmax)
        return spatial_softmax
