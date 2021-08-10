import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
from einops import rearrange, reduce, repeat

class ResNetFeats(nn.Module):
    def __init__(self, out_dim=256, output_raw=False, drop_dim=1, use_resnet18=False, pretrained=False):
        super(ResNetFeats, self).__init__()
        print('pretrain', pretrained)
        resnet = models.resnet18(pretrained=pretrained) if use_resnet18 else models.resnet50(pretrained=pretrained)
        self._features = nn.Sequential(*list(resnet.children())[:-drop_dim])
        self._output_raw = output_raw
        self._out_dim = 512 if use_resnet18 else 2048
        self._out_dim = int(self._out_dim / 2 ** (drop_dim - 2)) if drop_dim >= 2 else self._out_dim
        if not output_raw:
            self._nn_out = nn.Sequential(nn.Conv2d(self._out_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True), nn.Conv2d(out_dim, out_dim, 1))
            self._out_dim = out_dim
    
    def forward(self, inputs, depth=None):
        reshaped = len(inputs.shape) == 5
        x = inputs.reshape((-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]))

        out = self._features(x)
        out = self._nn_out(out) if not self._output_raw else out 
        NH, NW = out.shape[-2:]

        # reshape to proper dimension
        out = out.reshape((inputs.shape[0], inputs.shape[1], self._out_dim, NH, NW)) if reshaped else out
        if NH * NW == 1:
            out = out[:,:,:,0,0] if reshaped else out[:,:,0,0]
        return out

    @property
    def dim(self):
        return self._out_dim

class VGGFeats(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self._vgg = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:9])

        def _conv_block(in_dim, out_dim, N):
            # pool and coord conv
            cc = CoordConv(in_dim, out_dim, 2, stride=2, padding=0)
            n, a = nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
            ops = [cc, n, a]

            for _ in range(N):
                c = nn.Conv2d(out_dim, out_dim, 3, padding=1)
                n, a = nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
                ops.extend([c, n, a])
            return ops

        self._v1 = nn.Sequential(*_conv_block(128, 256, 3))
        self._v2 = nn.Sequential(*_conv_block(256, 512, 3))
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._out_dim = out_dim
        self._linear = nn.Sequential(nn.Linear(512, out_dim), nn.ReLU(inplace=True))

    def forward(self, x, depth=None):
        reshaped = len(x.shape) == 5
        in_x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])) if reshaped else x
        vis_feat = self._pool(self._v2(self._v1(self._vgg(in_x))))
        out_feat = self._linear(vis_feat[:,:,0,0])
        out_feat = out_feat.view((x.shape[0], x.shape[1], self._out_dim)) if reshaped else out_feat
        return out_feat

    @property
    def dim(self):
        return self._out_dim

class TemporalPositionalEncoding(nn.Module):
    """
    Modified PositionalEncoding from Pytorch Seq2Seq Documentation
    source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        print("Position embedding shape: {} \n".format(pe.shape))
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert len(x.shape) >= 3, "x requires at least 3 dims! (B, C, ..)"
        old_shape = x.shape
        x = x.reshape((x.shape[0], x.shape[1], -1)) # B,d,T*H*W
        x = x + self.pe[:,:,:x.shape[-1]]
        return self.dropout(x).reshape(old_shape)

class NonLocalLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feedforward_dim=512, dropout=0, temperature=None, causal=False, n_heads=1):
        super().__init__()
        assert feedforward_dim % n_heads == 0, "n_heads must evenly divide feedforward_dim"
        self._n_heads = n_heads
        self._temperature = temperature if temperature is not None else np.sqrt(in_dim)
        self._K = nn.Conv3d(in_dim, feedforward_dim, 1, bias=False)
        self._V = nn.Conv3d(in_dim, feedforward_dim, 1, bias=False)
        self._Q = nn.Conv3d(in_dim, feedforward_dim, 1, bias=False)
        self._out = nn.Conv3d(feedforward_dim, out_dim, 1)
        self._a1, self._drop1 = nn.ReLU(inplace=dropout==0), nn.Dropout3d(dropout)
        self._norm = nn.BatchNorm3d(out_dim)
        self._causal = causal
        self._skip = out_dim == in_dim

    def forward(self, inputs):
        # inputs shape: B, d, T, H, W
        K, Q, V = self._K(inputs), self._Q(inputs), self._V(inputs)
        B, C, T, H, W = K.shape 
        K, Q, V = [
            rearrange(t, 'B (head channel) T H W -> B head channel (T H W)', \
                head=self._n_heads, channel=int(C / self._n_heads))
            for t in (K, Q, V)]
        KQ = torch.einsum('bnci,bncj->bnij', K, Q) / self._temperature # B, heads, THW, THW
        if self._causal:
            mask = torch.tril(torch.ones((T,T))).to(KQ.device)
            mask = mask.repeat_interleave(H*W,0).repeat_interleave(H*W, 1) # -> (T*H*W, T*H*W)
            KQ = KQ + torch.log(mask).unsqueeze(0).unsqueeze(0) # -> (1, 1, T*H*W, T*H*W)
        attn = F.softmax(KQ, 3)
        V = torch.einsum('bncj,bnij->bnci', V, attn).reshape((B, C, T, H, W))
        out = inputs + self._drop1(self._a1(self._out(V))) if self._skip else self._drop1(self._a1(self._out(V)))
        return self._norm(out)

class TempConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feedforward_dim=512, dropout=0, temperature=None, causal=False, n_heads=None, k=5):
        super().__init__()
        self._causal, self._skip = causal, out_dim == in_dim
        frame_pad = int(k // 2)
        self._pad = k - 1 if causal else frame_pad
        self._c1, self._a1 = nn.Conv3d(in_dim, feedforward_dim, 1), nn.ReLU(inplace=True)
        self._c2, self._a2 = nn.Conv3d(feedforward_dim, feedforward_dim, k, padding=(self._pad, frame_pad, frame_pad)), nn.ReLU(inplace=True)
        self._c3, self._a3 = nn.Conv3d(feedforward_dim, out_dim, 1), nn.ReLU(inplace=dropout==0)
        self._drop = nn.Dropout3d(dropout)
        self._norm = nn.BatchNorm3d(out_dim)

    def forward(self, inputs):
        downsized = self._a1(self._c1(inputs))
        ff = self._a2(self._c2(downsized))
        ff = ff[:,:,:-self._pad] if self._causal else ff
        upsized = self._drop(self._a3(self._c3(ff)))
        out = inputs + upsized if self._skip else upsized
        return self._norm(out)
