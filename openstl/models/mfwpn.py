import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import numbers
from einops import rearrange
from openstl.modules import (ConvSC, GASubBlock)


class MFWPN_Model(nn.Module):
    def __init__(self, hid_S=2, hid_T=256, N_S=2, N_T=8, model_type='gsta',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(MFWPN_Model, self).__init__()
        T, C, H, W = 24, 2, 64, 80  # T is pre_seq_length
        H, W = 64, 80
        act_inplace = False
        self.enc = Encoder(spatio_kernel=spatio_kernel_enc, act_inplace=True)
        self.dec = Decoder(spatio_kernel=spatio_kernel_dec, act_inplace=True)
        self.hid_w = MidMetaNet(T*hid_S, hid_T, N_T, input_resolution=(64, 80), model_type=model_type,
                                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)    
        self.hid_tz = MidMetaNet(T*hid_S, hid_T, N_T, input_resolution=(64, 80), model_type=model_type,
                                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.channel_f = CAM(channel = 48)
        self.channel_i = CAM(channel = 48) 
        self.gate = nn.Tanh()
        self.ele_conv = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(48, 48, kernel_size=1, stride=1))
          
    def forward(self, x_raw, ele, **kwargs):
        B, T, C, H, W = x_raw.shape
        
        ele = ele[None, None, :, :]
        ele_h = ele
        ele = ele.repeat(B*T,1,1,1)
        x = x_raw.view(B*T, C, H, W)  
        wcs, tzcs = self.enc(x, ele)
        
        _, C_w, H_, W_ = wcs.shape
        _, C_tz, H_, W_ = tzcs.shape

        w = wcs.view(B, T, C_w, H_, W_)
        tz = tzcs.view(B, T, C_tz, H_, W_)

        hid_w = self.hid_w(w)  
        hid_tz = self.hid_tz(tz)
        #ele_h = ele_h.repeat(B, 48, 1, 1)
        ####time fusion
        hid_w = hid_w * self.channel_f(hid_tz) + self.channel_i(hid_tz) * self.gate(hid_tz)  
        hid_w = hid_w.reshape(B , T, C_w, H_, W_)
        hid_w = hid_w.reshape(B * T, C_w, H_, W_)
        
        Y = self.dec(hid_w)      
        Y = Y.reshape(B, T, 2, H, W)

        return Y
    
class SAM(nn.Module):
    def __init__(self, spatial_kernel=7):
        super(SAM, self).__init__()
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)      
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return spatial_out

class CAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAM, self).__init__()
   
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(       
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),   
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )        
        self.sigmoid = nn.Sigmoid()
                                   
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        eca = max_out + avg_out
        channel_out = self.sigmoid(eca)                           
        return channel_out
                                  
def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    """3D Encoder for SimVP"""
    def __init__(self, spatio_kernel, act_inplace=True):
        super(Encoder, self).__init__()  
        
        self.wscale = nn.Parameter(torch.ones(2))
        self.tzscale = nn.Parameter(torch.ones(2))
         ####Conv branch
        self.wce =  DetailFeatureExtraction(num_layers=1)
        self.tzce = DetailFeatureExtraction(num_layers=1)
        ####Self-attention branch
        self.wse = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False,
                                     LayerNorm_type='WithBias')
        
        self.tzse = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False,
                                     LayerNorm_type='WithBias')   
        self.sa_f = SAM()
        self.sa_i = SAM()
        self.gate_1 = nn.Sigmoid()
        self.gate_2 = nn.Tanh()
        ####ele
        self.ele_conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(2, 2, kernel_size=1, stride=1))
        
    def forward(self, x, ele):
        w = x[:, 0:2]
        tz = x[:, 2:4]
        
        wce_x = self.wce(w)
        wse_x = self.wse(w)
        wcs = wse_x * self.wscale[0] + wce_x * self.wscale[1]

        tzce_x = self.tzce(tz)
        tzse_x = self.tzse(tz)
        tzcs = tzse_x * self.tzscale[0] + tzce_x * self.tzscale[1]
        ####Fusion module
        tz_gate_2 = self.gate_2(tzcs)
        wcs = wcs * self.sa_f(tzcs) + self.sa_i(tzcs) * self.gate_2(tzcs) + wcs * self.gate_1(self.ele_conv(ele))

        return wcs, tzcs

class Decoder(nn.Module):
    """3D Encoder for SimVP"""
    def __init__(self, spatio_kernel, act_inplace=True):
        super(Decoder, self).__init__()
        
        self.dwscale = nn.Parameter(torch.ones(2))
        ####Conv branch
        self.dwce =  DetailFeatureExtraction(num_layers=1)                   
        ####Self-attention branch
        self.dwse = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False,
                                     LayerNorm_type='WithBias')
    def forward(self, wdcs):
        
        dwse_x = self.dwse(wdcs)
        dwce_x = self.dwce(wdcs)
        dwcs = dwse_x * self.dwscale[0] + dwce_x * self.dwscale[1]
        
        return dwcs

class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'
        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"
            
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)
    
class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]
        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1)) 
        self.enc_1 = nn.Sequential(*enc_layers)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc_1[i](z)
        return z

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out 
##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
##########################################################################
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            #nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        
    def forward(self, z1, z2):
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        
        self.shffleconv = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.fusion = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    
    def forward(self, x):
        z1, z2 = self.separateFeature(self.shffleconv(x))
        for layer in self.net: 
            z1, z2 = layer(z1, z2)
        z_out = self.fusion(torch.cat((z1, z2), dim=1))
        return z_out