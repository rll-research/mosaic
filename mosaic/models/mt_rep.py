import copy 
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mosaic.models import get_model
from mosaic.models.discrete_logistic import DiscreteMixLogistic
from mosaic.models.rep_modules import BYOLModule, ContrastiveModule
from mosaic.models.basic_embedding import TemporalPositionalEncoding 
from einops import rearrange, repeat, parse_shape
from collections import OrderedDict
 
class _StackedAttnLayers(nn.Module):
    """
    Returns all intermediate-layer outputs, in case we want to re-use features + add more losses later
                -> default sharing norms now!
    Demo and obs share the same batchnorm at every layer;
    Construct a stack of layers at the same time, note this would give us more control 
    over how the attended embedding of demonstration and observation is combined.
    attention matrics here are even _smaller_ if we set fuse_starts to >0
    Args to note
    - fuse_starts: counts for how many layers demo & obs should attend to itself, independent of each other 
     
    """
    def __init__(
        self, 
        in_dim, 
        out_dim,
        n_layers=3, 
        fuse_starts=0,
        demo_ff_dim=128, 
        obs_ff_dim=128, 
        dropout=0, 
        temperature=None, 
        causal=False, 
        n_heads=4, 
        demo_T=4, 
        ):
        super().__init__()
        assert demo_ff_dim % n_heads == 0, "n_heads must evenly divide feedforward_dim"
        self._n_heads       = n_heads
        self._demo_ff_dim   = demo_ff_dim
        self._obs_ff_dim    = obs_ff_dim 
        self._temperature   = temperature if temperature is not None else np.sqrt(in_dim)

        self._obs_Qs, self._obs_Ks, self._obs_Vs = [
            nn.Sequential( *[nn.Conv3d(in_dim, obs_ff_dim, 1, bias=False) for _ in range(n_layers)])
            for _ in range(3)]
        self._obs_Outs     =  nn.Sequential( *[nn.Conv3d(obs_ff_dim, out_dim, 1, bias=False) for _ in range(n_layers)])
        self._obs_a1s      =  nn.Sequential( *[nn.ReLU(inplace=dropout==0) for _ in range(n_layers)]   )
        self._obs_drop1s   =  nn.Sequential( *[nn.Dropout3d(dropout)       for _ in range(n_layers)]   ) 
        
        self._demo_Qs, self._demo_Ks, self._demo_Vs = [
            nn.Sequential( *[nn.Conv3d(in_dim, demo_ff_dim, 1, bias=False) for _ in range(n_layers)])
            for _ in range(3)]
        self._demo_Outs    =  nn.Sequential( *[nn.Conv3d(demo_ff_dim, out_dim, 1, bias=False) for _ in range(n_layers)])
        self._demo_a1s     =  nn.Sequential( *[nn.ReLU(inplace=dropout==0) for _ in range(n_layers)]   )
        self._demo_drop1s  =  nn.Sequential( *[nn.Dropout3d(dropout)       for _ in range(n_layers)]   ) 
        self._norms    =  nn.Sequential( *[nn.BatchNorm3d(out_dim)     for _ in range(n_layers)]   )
            
        self._n_layers     =  n_layers
        self._fuse_starts  =  fuse_starts
        print("A total of {} layers, cross demo-obs attention starts at layer idx {}".format(n_layers, fuse_starts))
        self._causal       =  causal # keep causal only as an option for demo attention
        self._skip         = out_dim == in_dim
        self._demo_T       = demo_T 
    
    def forward(self, inputs):
        """"""
        B, d, T, H, W       = inputs.shape 
        obs_T               = T - self._demo_T                            # obs_T could be as small as 1
        out_dict            = dict()
        for i in range(self._n_layers):
            demo_ly_in, obs_ly_in = inputs.split([self._demo_T, obs_T], dim=2)
            # -> (B, d, demo_T, H, W), (B, d, obs_T, H, W)
            ## process demo first 
            demo_q, demo_k, demo_v =  [
                rearrange( conv[i](demo_ly_in), 'B (head ch) T H W -> B head ch (T H W)', head=self._n_heads)
                for conv in [self._demo_Qs, self._demo_Ks, self._demo_Vs]
            ]
            # if self.query_task:
            #     demo_q = torch.cat((self.demo_q))
            a1, drop1 = [mod[i] for mod in [self._demo_a1s, self._demo_drop1s ] ]
            B, head, ch, THW = demo_q.shape
            ff_dim = head * ch
            demo_kq = torch.einsum('bnci,bncj->bnij', demo_k, demo_q) / self._temperature # B, heads, T_demo*HW, T_demo*HW
            if self._causal:
                mask = torch.tril(torch.ones((self._demo_T, self._demo_T))).to(demo_kq.device)
                mask = mask.repeat_interleave(H*W,0).repeat_interleave(H*W, 1) # -> (T*H*W, T*H*W)      
                demo_kq = demo_kq + torch.log(mask).unsqueeze(0).unsqueeze(0) # -> (1, 1, T*H*W, T*H*W)
            demo_attn = F.softmax(demo_kq, 3)
            demo_v = torch.einsum('bncj,bnij->bnci', demo_v, demo_attn)
            demo_out = self._demo_Outs[i](
                rearrange(demo_v, 'B head ch (T H W) -> B (head ch) T H W', T=self._demo_T, H=H, W=W)
                )
            demo_out = demo_out + drop1(a1(demo_out)) if self._skip else drop1(demo_out)
            

            ## now, repeat demo's K and V for obs. NOTE: **brought the T dimension forward**
            obs_q, obs_k, obs_v = [
                rearrange( conv[i](obs_ly_in), 
                    'B (head ch) obs_T H W -> B obs_T head ch (H W)', head=self._n_heads)
                    for conv in [self._obs_Qs, self._obs_Ks, self._obs_Vs]
                ] 
            a1, drop1 = [mod[i] for mod in [self._obs_a1s, self._obs_drop1s ] ]
            if i >= self._fuse_starts: 
                rep_k, rep_v = [
                    repeat(rep, 'B head ch THW -> B obs_T head ch THW', obs_T=obs_T) for rep in [demo_k, demo_v]]
                # only start attending to demonstration a few layers later
                cat_k = torch.cat([rep_k, obs_k], dim=4) # now cat_k is B, T, head, ch, (4+1)HW
                cat_v = torch.cat([rep_v, obs_v], dim=4) 
            else:
                cat_k, cat_v = obs_k, obs_v  # only attend to observation selves
            obs_kq = torch.einsum( 
                'btnci,btncj->btnij', cat_k, obs_q) / self._temperature # B, obs_T, heads, (1+T_demo)*HW, 1*HW
            assert obs_kq.shape[-2] == (1+self._demo_T)* H * W or \
                    obs_kq.shape[-2] == H*W 
            # no causal mask is needed
            obs_attn = F.softmax(obs_kq, dim=4)
            obs_v = torch.einsum( 'btncj,btnji->btnci', cat_v, obs_attn )
            
            obs_out = self._obs_Outs[i](
                rearrange(obs_v, 'B T heads ch (H W) -> B (heads ch) T H W', H=H, W=W, T=obs_T)
                )
            obs_ly_in = obs_ly_in + drop1(a1(obs_out)) if self._skip else drop1(a1(obs_out))
            
            inputs = self._norms[i](torch.cat([demo_ly_in, obs_ly_in], dim=2))
            out_dict['out_%s'%i] = inputs
        out_dict['last'] = inputs
        return out_dict

    def freeze_layer(self, i):
        count = 0
        for conv in [self._demo_Qs, self._demo_Ks, self._demo_Vs] + [self._obs_Qs, self._obs_Ks, self._obs_Vs]:
            to_freeze = conv[i]
            for param in to_freeze.parameters():
                if param.requires_grad:
                    param.requires_grad = False 
                    count += np.prod(param.shape)
        for mod in [self._demo_a1s, self._demo_drop1s, self._obs_a1s, self._obs_drop1s, self._norms]:
            to_freeze = mod[i]
            for param in to_freeze.parameters():
                if param.requires_grad:
                    param.requires_grad = False 
                    count += np.prod(param.shape)
        return count 

class _TransformerFeatures(nn.Module):
    """
    Transformer-like module for computing self-attention on convolution features
    """
    def __init__(
        self, latent_dim, demo_T=4, dim_H=7, dim_W=12, embed_hidden=256, dropout=0.2, n_attn_layers=2,
        pos_enc=True, attn_heads=4, attn_ff=128, just_conv=False,
        pretrained=True, img_cfg=None, drop_dim=2, 
        causal=True, attend_demo=True, demo_out=True,
        fuse_starts=0):
        super().__init__()

        flag, drop_dim = img_cfg.network_flag, img_cfg.drop_dim
        self.network_flag = flag

        assert flag == 0 , "flag number %s not supported!" % flag
        self._img_encoder = get_model('resnet')(
            output_raw=True, drop_dim=drop_dim, use_resnet18=True, pretrained=img_cfg.pretrained)
        if drop_dim == 2:
            conv_feature_dim = 512
        elif drop_dim == 3:
            conv_feature_dim = 256
        else:
            raise NotImplementedError
        
        self._attn_layers = _StackedAttnLayers(
            in_dim=conv_feature_dim, out_dim=conv_feature_dim, n_layers=n_attn_layers,
            demo_ff_dim=attn_ff, obs_ff_dim=attn_ff, dropout=dropout,
            causal=causal, n_heads=attn_heads, demo_T=demo_T, fuse_starts=fuse_starts,
        )

        self._pe = TemporalPositionalEncoding(conv_feature_dim, dropout) if pos_enc else None
        self.demo_out = demo_out
        in_dim = conv_feature_dim * dim_H * dim_W
        print("Linear embedder has input dim: {}x{}x{}={} ".format(
            conv_feature_dim, dim_H, dim_W, in_dim))

        self._linear_embed = nn.Sequential(
            nn.Linear(in_dim, embed_hidden),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(embed_hidden, latent_dim))

    def forward(self, images, context):
        assert len(images.shape) == 5, "expects [B, T, 3, height, width] tensor!"
        obs_T, demo_T = images.shape[1], context.shape[1]
        out_dict = OrderedDict()

        network_fn = self._resnet_features if self.network_flag == 0 else self._impala_features

        im_in                     = torch.cat((context, images), 1)
        im_features, no_pe_img_features = network_fn(im_in)
        out_dict['img_features']  = no_pe_img_features # B T d H W
        out_dict['img_features_pe'] = rearrange(im_features, 'B d T H W -> B T d H W')
        # print(no_pe_img_features.shape)
        attn_out                  =  self._attn_layers( im_features )
        attn_features             =  attn_out['last']    # just use this for now, try other stuff later

        sizes                     = parse_shape(attn_features, 'B _ T _ _')
        features                  = rearrange(attn_features, 'B d T H W -> B T d H W', **sizes)
        out_dict['attn_features'] = features
        out_dict['demo_features'], out_dict['obs_features'] = \
            features.split([demo_T, obs_T], dim=1)
        # could also try do repre. on all intermediate layers too
        for k, v in attn_out.items():
            if k != 'last' and v.shape == attn_features.shape:
                reshaped = rearrange(v, 'B d T H W -> B T d H W', **sizes)
                out_dict['attn_'+k] = reshaped
                normalized = F.normalize(
                    self._linear_embed(rearrange(reshaped, 'B T d H W -> B T (d H W)')), dim=2)
                out_dict['attn_'+k+'_demo'], out_dict['attn_'+k+'_img'] = normalized.split([demo_T, obs_T], dim=1)

        out_dict['linear_embed']  = self._linear_embed(rearrange(features, 'B T d H W -> B T (d H W)'))
        normalized                = F.normalize(out_dict['linear_embed'], dim=2)
        out_dict['normed_linear_embed'] = normalized

        demo_embed, img_embed     = normalized.split([demo_T, obs_T], dim=1)
        out_dict['demo_embed']    = demo_embed
        out_dict['demo_mean']     = torch.mean(demo_embed, dim=1)
        out_dict['img_embed']     = img_embed

        # NOTE(0427) this should always have length demo_T + obs_T now
        return out_dict

    def _resnet_features(self, x):
        if self._pe is None:
            return self._img_encoder(x)
        features = self._img_encoder(x) # x is B, T, ch, h, w -> B, T, d, H, W
        pe_features = self._pe(features.transpose(1,2))
        return pe_features, features # B T d H W

    def _impala_features(self, x):
        sizes = parse_shape(x, 'B T _ _ _') # batch_size, concat_size = x.shape[0], x.shape[1] # B, T_c+T_im, 3, height, width
        x = rearrange(x, 'B T ch height width -> (B T) ch height width')
        features = self._img_encoder(x) # B*T, d=256, H=6, W=9
        features = rearrange(features, '(B T) d H W -> B d T H W', **sizes)
        if self._pe is None:
            return features
        pe_features = self._pe(features)
        no_pe_features = rearrange(features, 'B d T H W -> B T d H W')

        return pe_features, no_pe_features

class _DiscreteLogHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures, const_var=True, sep_var=False):
        super().__init__()
        assert n_mixtures  >= 1, "must predict at least one mixture!"
        self._n_mixtures   =  n_mixtures
        self._dist_size    =  torch.Size((out_dim, n_mixtures))
        self._mu           =  nn.Linear(in_dim, out_dim * n_mixtures)
        self._logit_prob   =  nn.Linear(in_dim, out_dim * n_mixtures) if n_mixtures > 1 else None
        if const_var:
            ln_scale       = torch.randn(out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        if sep_var:
            ln_scale       = torch.randn((out_dim, n_mixtures), dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        if not (const_var or sep_var):
            self._ln_scale = nn.Linear(in_dim, out_dim * n_mixtures)

    def forward(self, x): # x has shape B T d
        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))

        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(x).reshape((x.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            if len(ln_scale.shape) == 1:
                ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)
                #(1, 1, 8, 1) -> (B T, dist_size[0], dist_size[1]) i.e. each mixture has the **same** constant variance
            else: # the sep_val case:
                ln_scale = repeat(ln_scale, 'out_d n_mix -> B T out_d n_mix', B=x.shape[0], T=x.shape[1])

        logit_prob = self._logit_prob(x).reshape(mu.shape) if self._n_mixtures > 1 else torch.ones_like(mu)
        return (mu, ln_scale, logit_prob)

class VideoImitation(nn.Module):
    """ The imitation policy model  """
    def __init__(
        self,
        latent_dim,
        height=120,
        width=160,
        demo_T=4,
        obs_T=6,
        dim_H=7,
        dim_W=12,
        action_cfg=None,
        attn_cfg=None,
        sdim=8,
        concat_state=False,
        atc_config=None,
        curl_config=None,
        concat_demo_head=False,
        concat_demo_act=False,
        demo_mean=0,
        byol_config=dict(),
        simclr_config=dict(),
        ):
        super().__init__()

        self._embed = _TransformerFeatures(latent_dim=latent_dim, demo_T=demo_T, dim_H=dim_H, dim_W=dim_W, **attn_cfg)
        self._target_embed = copy.deepcopy(self._embed)
        self._target_embed.load_state_dict(self._embed.state_dict())
        for p in self._target_embed.parameters():
            p.requires_grad = False             # only update with soft param update!

        # one auxillary module calculate multiple losses
        # create a dummy input to calculate feature dimensions here
        with torch.no_grad():
            x = torch.zeros((1, demo_T+obs_T, 3, height, width))

            _out = self._embed(images=x[:, :demo_T], context=x[:, demo_T:])
            img_feats = _out['img_features']
            print("Image feature dimensions: {}".format(img_feats.shape))
            _, _, img_conv_dim, _, _ = img_feats.shape
            attn_feats = _out['attn_features']
            _, _, attn_conv_dim, _, _ = attn_feats.shape
            assert img_feats.shape[1] == attn_feats.shape[1] or img_feats.shape[1] == attn_feats.shape[1] + demo_T # should both be B, demo_T+obs, _, _, _
            img_feat_dim = np.prod(img_feats.shape[2:]) # should be d*H*W
            attn_feat_dim = np.prod(attn_feats.shape[2:])
            if img_feat_dim != attn_feat_dim:
                print("Warning! pre and post attn features have different shapes:",
                    img_feat_dim, attn_feat_dim)
        self._demo_T = demo_T

        self._byol = BYOLModule(
            embedder=self._target_embed,
            img_feat_dim=img_feat_dim, attn_feat_dim=attn_feat_dim,
            img_conv_dim=img_conv_dim, attn_conv_dim=attn_conv_dim, **byol_config)
        self._simclr = ContrastiveModule(
            embedder=self._target_embed,
            img_feat_dim=img_feat_dim, attn_feat_dim=attn_feat_dim,
            img_conv_dim=img_conv_dim, attn_conv_dim=attn_conv_dim, **simclr_config)

        self._concat_state = concat_state
        # action processing
        assert action_cfg.n_mixtures >= 1, "must predict at least one mixture!"
        self.concat_demo_head = concat_demo_head
        self.concat_demo_act  = concat_demo_act
        assert not (concat_demo_head and concat_demo_act), 'Only support one concat type'
        print("Concat-ing embedded demo to action head? {}, to distribution head? {}".format(concat_demo_act, concat_demo_head))

        # NOTE(Mandi): reduced input dimension size  from previous version! hence maybe try widen/add more action layers
        ac_in_dim = int(latent_dim + float(concat_demo_act) * latent_dim + float(concat_state) * sdim)
        
        if action_cfg.n_layers == 1:
            self._action_module = nn.Sequential(nn.Linear(ac_in_dim, action_cfg.out_dim), nn.ReLU())
            self._inv_model     = nn.Sequential(nn.Linear(2*ac_in_dim, action_cfg.out_dim), nn.ReLU())
        elif action_cfg.n_layers == 2:
            self._action_module = nn.Sequential(
                nn.Linear(ac_in_dim, action_cfg.hidden_dim), nn.ReLU(),
                nn.Linear(action_cfg.hidden_dim, action_cfg.out_dim), nn.ReLU()
                )
            self._inv_model     = nn.Sequential(
                nn.Linear(2 * ac_in_dim, action_cfg.hidden_dim), nn.ReLU(),
                nn.Linear(action_cfg.hidden_dim, action_cfg.out_dim), nn.ReLU()
                )
        else:
            raise NotImplementedError

        head_in_dim = int(action_cfg.out_dim + float(concat_demo_head) * latent_dim) 
        self._action_dist = _DiscreteLogHead(
            in_dim=head_in_dim,
            out_dim=action_cfg.adim,
            n_mixtures=action_cfg.n_mixtures,
            const_var=action_cfg.const_var,
            sep_var=action_cfg.sep_var
            )
        self.demo_mean = demo_mean


        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Total params in Imitation module:', params)


    def get_action(self, embed_out, ret_dist=True, states=None):
        """directly modifies output dict to put action outputs inside"""
        out = dict()
        ## single-head case
        demo_embed, img_embed   = embed_out['demo_embed'], embed_out['img_embed']
        assert demo_embed.shape[1] == self._demo_T
        obs_T = img_embed.shape[1]
        ac_in                   = img_embed
        if self.demo_mean:
            demo_embed          = torch.mean(demo_embed, dim=1)
        else:
            demo_embed          = demo_embed[:, -1, :] # only take the last image, should alread be attended tho
        demo_embed              = repeat(demo_embed, 'B d -> B ob_T d', ob_T=obs_T)

        if self.concat_demo_act: # for action model 
            ac_in                 = torch.cat((img_embed, demo_embed), dim=2)
            ac_in               = F.normalize(ac_in, dim=2)
        ac_in                   = torch.cat((ac_in, states), 2) if self._concat_state else ac_in
        # predict behavior cloning distribution
        ac_pred                 = self._action_module(ac_in)
        if self.concat_demo_head:
            ac_pred             = torch.cat((ac_pred, demo_embed), dim=2)
            ac_pred             = F.normalize(ac_pred, dim=2) # maybe better to normalize here

        mu_bc, scale_bc, logit_bc = self._action_dist(ac_pred)
        out['bc_distrib']       = DiscreteMixLogistic(mu_bc, scale_bc, logit_bc) \
            if ret_dist else (mu_bc, scale_bc, logit_bc)
        out['demo_embed']       = demo_embed
        ## multi-head case? maybe register a name for each action head
        return out

    def forward(
        self,
        images,
        context,
        states=None,
        ret_dist=True,
        eval=False,
        images_cp=None,
        context_cp=None,
        actions=None,
        ):
        B, obs_T, _, height, width = images.shape
        demo_T = context.shape[1]
        if not eval:
            assert images_cp is not None, 'Must pass in augmented version of images'
        embed_out = self._embed(images, context)
        out = self.get_action(embed_out=embed_out, ret_dist=ret_dist, states=states)

        if eval:
            return out # NOTE: early return here to do less computation during test time
 
        # run frozen transformer on augmented images
        embed_out_target = self._target_embed(images_cp, context_cp)

        byol_out_dict = self._byol(embed_out, embed_out_target)
        for k, v in byol_out_dict.items():
            assert 'byol' in k
            out[k] = v

        simclr_out_dict = self._simclr(embed_out, embed_out_target)
        for k, v in simclr_out_dict.items():
            assert 'simclr' in k
            out[k] = v

        # run inverse model
        demo_embed, img_embed        = out['demo_embed'], embed_out['img_embed'] # both are (B ob_T d)
        inv_in                       = torch.cat((img_embed[:,:-1], img_embed[:,1:]), 2)  # B, T_im-1, d * 2
        if self.concat_demo_act:
            inv_in   = torch.cat(
                (
                F.normalize(torch.cat((img_embed[:, :-1], demo_embed[:,:-1]), dim=2), dim=2),
                F.normalize(torch.cat((img_embed[:,  1:], demo_embed[:,:-1]), dim=2), dim=2),
                ),
                dim=2)
            # print(inv_in.shape)
        inv_pred                     = self._inv_model(inv_in)
        if self.concat_demo_head:
            inv_pred                 = torch.cat((inv_pred, demo_embed[:, :-1]), dim=2)
            inv_pred                 = F.normalize(inv_pred, dim=2) # maybe better to normalize here

        mu_inv, scale_inv, logit_inv = self._action_dist(inv_pred)
        out['inverse_distrib']       = DiscreteMixLogistic(mu_inv, scale_inv, logit_inv) \
                                        if ret_dist else (mu_inv, scale_inv, logit_inv)

        return out

    def momentum_update(self, frac):
        self._byol.update_mom(frac)
        return

    def soft_param_update(self):
        self._byol.soft_param_update()
        tau = 1 - self._byol.mom
        for param, target_param in zip(self._embed.parameters(), self._target_embed.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        return

    def freeze_attn_layers(self, n_layers=2):
        assert n_layers <= self._embed._attn_layers._n_layers, 'Attention only has %s layers' % self._embed._attn_layers._n_layers
        count = 0
        for n in range(n_layers):
            num_frozen = self._embed._attn_layers.freeze_layer(n)
            count += num_frozen
        print("Warning! Freeze the _First_ {} layers of attention! A total of {} params are frozen \n".format(n_layers, count))

    def freeze_img_encoder(self):
        count = 0
        for p in self._embed._img_encoder.parameters():
            if p.requires_grad:
                p.requires_grad = False
                count += np.prod(p.shape)
        print("Warning! Freeze %s parameters in the image encoder \n" % count)

    def restart_action_layers(self):
        count = 0
        for module in [self._action_module, self._action_dist, self._inv_model]:
            for layer in module.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                for p in layer.parameters():
                    count += np.prod(p.shape) if p.requires_grad else 0
        print("Re-intialized a total of %s parameters in action MLP layers" % count)

    def skip_for_eval(self):
        """ skip module inferences during evaluation"""
        self._byol = None
        self._simclr = None
        self._inv_model = None
        self._target_embed = None

    def pretrain_img_encoder(self):
        """Freeze everything except for the image encoder + attention layers"""
        count = 0
        for p in zip(
            self._action_module.parameters(),
            self._inv_model.parameters(),
            self._embed._linear_embed.parameters()
            ):
            p.requires_grad = False
            count += np.prod(p.shape)
        print("Freezing action, inv, linear embed {} layer parameters".format(count))
