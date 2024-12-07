import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import math

import distutils.version
import random
from models.loss import VM_diffeo_loss, NCC
from models.GP_layer import GPlayer

from einops import rearrange
from einops.layers.torch import Rearrange


MIN_NUM_PATCHES = 16

table = [
            # t, c, n, s, SE
            [1,  24,  2, 1, 0],
            [4,  48,  4, 2, 0],
            [4,  64,  4, 2, 0],
            [4, 128,  6, 2, 1],
            #[6, 160,  9, 1, 1],
            #[6, 256, 15, 2, 1],
        ]


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

def grid_sample(*args,**kwargs):
    if distutils.version.LooseVersion(torch.__version__) < distutils.version.LooseVersion("1.3.0"):
        return torch.nn.functional.grid_sample(*args,**kwargs)
    else:
        return torch.nn.functional.grid_sample(*args,**kwargs,align_corners=True)

def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)

# Feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim / heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_qkv_h = nn.Linear(dim, dim * 3)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
        self.reattn_norm = nn.Sequential(
                            Rearrange('b h i j -> b i j h'),
                            nn.LayerNorm(heads),
                            Rearrange('b i j h -> b h i j'))

        self.elu = nn.ELU(alpha=1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),)
    
    def forward(self, x, hidden):
        b, n, _, h = *x.shape, self.heads
        qkv = torch.add(self.to_qkv(x), self.to_qkv_h(hidden)).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        out = self.linear_attn(self._elu(q), self._elu(k), v)
        # out = self.re_attn(q, k, v)
        # out = self.softmax_attn(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

    def _elu(self, x):
        return self.elu(x) + 1
    
    def softmax_attn(self, q, k, v):
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out

    def linear_attn(self, q, k, v):
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out

    def re_attn(self, q, k, v):
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = torch.einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)
        return torch.einsum('b h i j, b h j d -> b h i d', attn, v)


class Encoder1DBlock(nn.Module):
    def __init__(self, input_shape, heads, mlp_dim, dtype=torch.float32, dropout_rate=0.1, attention_dropout_rate=0.1, deterministic=True, rnn_layer = True):
        super().__init__()

        self.rnn_layer = rnn_layer
        self.layer_norm_input = nn.LayerNorm(input_shape)
        self.layer_norm_hidden = nn.LayerNorm(input_shape)
        self.layer_norm_out = nn.LayerNorm(input_shape)

        self.attention = MultiHeadDotProductAttention(input_shape, heads = heads)
        self.mlp = FeedForward(input_shape, mlp_dim, dropout_rate)
        self.drop_out_attention  = nn.Dropout(attention_dropout_rate)

    def forward(self, x, h):
        
        residual_x, residual_h = x, h
        attn = self.attention(self.layer_norm_input(x), self.layer_norm_hidden(h))
        x = self.drop_out_attention(attn) + residual_x
        
        residual_x = x
        x = self.layer_norm_out(x)
        x = self.mlp(x)
        x += residual_x

        return x, attn + residual_h

class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding layers
        """
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)

        return y

    
class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out    

class Encoder(nn.Module):
    def __init__(self, input_shape, num_layers, heads, mlp_dim, bidirection, dropout_rate=0.1):
        super(Encoder, self).__init__()
        # encoder blocks
        self.dropout = nn.Dropout(dropout_rate)
        self.layers_forward = nn.ModuleList([])
        self.bidirection = bidirection
        if bidirection:
            self.layers_backward = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers_forward.append(nn.ModuleList([Encoder1DBlock(input_shape, heads, mlp_dim)]))
            if bidirection:
                self.layers_backward.append(nn.ModuleList([Encoder1DBlock(input_shape, heads, mlp_dim)]))

    def forward(self, x, hidden, spatial_pos_embedding, temporal_pos_embedding):
        x_forward  = x.clone()
        x_backward = x.clone()
        length = x.shape[1]

        pos_embedding = torch.einsum('b i n d, b n j d -> b i j d', temporal_pos_embedding[:, :, None, ...], spatial_pos_embedding[:, None, ...])
        for idx in range(len(self.layers_forward)):
            x = x + pos_embedding
            hidden_forward = hidden.clone()
            hidden_backward = hidden.clone()
            for i in range(length):
                f_forward, hidden_forward = self.layers_forward[idx][0](self.dropout(x[:, i]), 
                                                                        self.dropout(hidden_forward + pos_embedding[:, idx, ...]))
                x_forward[:, i] = f_forward
                if self.bidirection:
                    f_backward, hidden_backward = self.layers_backward[idx][0](self.dropout(x[:, length-i-1]), 
                                                                               self.dropout(hidden_backward + pos_embedding[:, length-i-1, ...]))
                    x_backward[:, length-i-1] = f_backward
                else:
                    x_backward = 0

            x = x_forward + x_backward
        return x


class DiffeomorphicTransform(nn.Module):
    def __init__(self, size, mode='bilinear', time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode
        self.time_step = time_step

    def forward(self, velocity):
        flow = velocity / (2.0 ** self.time_step)

        # 1.0 flow
        for _ in range(self.time_step):
            new_locs = self.grid + flow
            shape = flow.shape[2:]

            # Need to normalize grid values to [-1, 1] for resampler
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
            
            if len(shape) == 2:
                new_locs = new_locs.permute(0, 2, 3, 1)
                new_locs = new_locs[..., [1, 0]]

            elif len(shape) == 3:
                new_locs = new_locs.permute(0, 2, 3, 4, 1)
                new_locs = new_locs[..., [2, 1, 0]]

            flow = flow + torch.nn.functional.grid_sample(flow, new_locs, align_corners=True, mode=self.mode)
        
        return flow


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return torch.nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Lagrangian_flow(nn.Module):
    """
    [Lagrangian_flow] is a class representing the computation of Lagrangian flow (v12, v13, v14, ...) from inter frame
    (INF) flow filed (u12, u23, u34, ...)
    v12 = u12
    v13 = v12 + u23 o v12 ('o' is a warping)
    v14 = v13 + u34 o v13
    ...
    """

    def __init__(self, vol_size):
        """
        Instiatiate Lagrangian_flow layer
            :param vol_size: volume size of the atlas
        """
        super(Lagrangian_flow, self).__init__()

        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, inf_flow, forward_flow=True):
        """
        Pass input x through forward once
            :param inf_flow: inter frame motion field
        """
        shape = inf_flow.shape
        seq_len = shape[2]
        lag_flow = torch.zeros(shape, device=inf_flow.device)
        lag_flow[:, :, 0] = inf_flow[:, :, 0]
        # lag_flow[0, ::] = inf_flow[0,::]
        for k in range (1, seq_len):
            if forward_flow:
                src = lag_flow[:, :, k-1].clone()
                sum_flow = inf_flow[:, :, k:k+1]
            else:         
                src = inf_flow[:, :, k]
                sum_flow = lag_flow[:, :, k-1:k]

            src_x = src[:, 0, ...].unsqueeze(1)
            src_y = src[:, 1, ...].unsqueeze(1)

            lag_flow_x = self.spatial_transform(src_x, sum_flow.squeeze(2))
            lag_flow_y = self.spatial_transform(src_y, sum_flow.squeeze(2))
            lag_flow[:, :, k] = sum_flow.squeeze(2) + torch.cat((lag_flow_x, lag_flow_y), dim=1)

        return lag_flow


class GPTrack2D(nn.Module):
    """ Vision Transformer """
    def __init__(self, image_size, patch_size, depth, length, heads, int_steps=7, mlp_dim=None, channels=2, dropout=0., emb_dropout=0, bidirection=True):
        super(GPTrack2D, self).__init__()

        """
        Instiatiate lagrangian_motion_estimate_net model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
            :param int_steps: the number of integration steps
        """

        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        hidden_dim = patch_size ** 2
        if mlp_dim is None:
            mlp_dim = hidden_dim
        else:
            hidden_dim = mlp_dim
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'
        
        vol_size = (image_size, image_size)
        dim = len(vol_size)
        enc_nf = [16, 32, mlp_dim, mlp_dim]
        dec_nf = [mlp_dim, 32, 32, 32, 16, 2]
        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow_mean = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        self.flow_log_sigma = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        self.alpha = 5
        self.beta = 1

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow_mean.weight = nn.Parameter(nd.sample(self.flow_mean.weight.shape))
        self.flow_mean.bias = nn.Parameter(torch.zeros(self.flow_mean.bias.shape))
        self.flow_log_sigma.weight = nn.Parameter(nd.sample(self.flow_log_sigma.weight.shape))
        self.flow_log_sigma.bias = nn.Parameter(torch.zeros(self.flow_log_sigma.bias.shape))

        self.embedding = nn.Conv2d(channels, hidden_dim, patch_size, patch_size)

        # positional embedding
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, 128, hidden_dim))
        
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))
        
        self.middle_conv_block = conv_block(dim, enc_nf[-1], enc_nf[-1])

        # transformer feature encoding
        self.transformer = Encoder(hidden_dim, depth, heads, mlp_dim, bidirection, dropout_rate = dropout)

        # velocity filed decoding
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5
        self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1)) # Full size conv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # setting the disffeomorph transform
        self.diffeomorph_transform = DiffeomorphicTransform(size=vol_size, mode='bilinear',time_step=int_steps)
        self.spatial_transform = SpatialTransformer(vol_size)

        # setting the Lagrangian flow
        self.lag_flow = Lagrangian_flow(vol_size)
        self.lag_regular = True

        self.gplayer = GPlayer()

        # feature dropout
        self.dropout = nn.Dropout(emb_dropout)

        # define loss
        self.velocity_loss = VM_diffeo_loss(image_sigma=0.02, prior_lambda=10, flow_vol_shape=vol_size)
        self.recon_loss = NCC()


    def forward(self, X, hidden, X_transformd=None, train=True):
        src_X = X[:,:-1]
        tgt_X = X[:,1:]
        cat_X = torch.cat((src_X, tgt_X), dim=2)

        b, t, _, _, _ = cat_X.shape
        input = rearrange(cat_X, 'b t c h w  -> (b t) c h w')

        x_enc = [input]
        for enc_layer in self.enc:
            x_enc.append(enc_layer(x_enc[-1]))
        
        emb = x_enc[-1]
        _, c, p_h, p_w = emb.shape
        emb = rearrange(emb, '(b t) c h w  -> b t c (h w)', b=b, t=t).transpose(-1, -2)

        sample_rate = 1
        if train:
            start = random.randint(0, (128 - sample_rate * t))
        else:
            start = 0

        # transformer
        feature = self.transformer(emb, hidden, self.spatial_pos_embedding, 
                                   self.temporal_pos_embedding[:, start : start + sample_rate * t : sample_rate])
        D = (torch.einsum('b i d, b j d -> b i j', self.temporal_pos_embedding[:, start : start + sample_rate * t : sample_rate], 
                                                   self.temporal_pos_embedding[:, start : start + sample_rate * t : sample_rate]) *  (c ** -0.5)).repeat(b, 1, 1)
        Z = self.gplayer(D, feature)
        Z = rearrange(feature, 'b t (h w) c  -> (b t) c h w', h=p_h, w=p_w)
        y = self.middle_conv_block(Z) + Z
        
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        y = self.upsample(y)
        y = torch.cat([y, x_enc[0]], dim=1)
        y = self.dec[5](y)
        
        flow_mean = self.flow_mean(y)
        flow_log_sigma = self.flow_log_sigma(y)

        # reparamterize
        std = torch.exp(0.5 * flow_log_sigma)
        vts = flow_mean + std * torch.rand_like(std)
        vts = rearrange(vts, '(b t) c h w  -> b t c h w', b=b, t=t)
        
        inf_flow_all = [ None for i in range(t) ]
        neg_inf_flow_all = [ None for i in range(t) ]
        inf_regsiter = [ None for i in range(t) ]
        neg_inf_regsiter = [ None for i in range(t) ]
        training_loss = 0

        for i in range(t):
            vt = vts[:, i]

            # bi-directional INF flows
            inf_flow = self.diffeomorph_transform(vt)
            neg_inf_flow = self.diffeomorph_transform(-vt)
            inf_flow_all[i] = inf_flow
            neg_inf_flow_all[i] = neg_inf_flow
            
            flow_param = torch.cat((flow_mean, flow_log_sigma), dim=1)
            # Image warping
            src, tgt = src_X[:, i], tgt_X[:, i]
            y_src = self.spatial_transform(src, inf_flow)
            y_tgt = self.spatial_transform(tgt, neg_inf_flow)
            inf_regsiter[i] = y_src
            neg_inf_regsiter[i] = y_tgt

            if train:
                src_X_t = X_transformd[:,:-1]
                tgt_X_t = X_transformd[:,1:]

                src_t, tgt_t = src_X_t[:, i], tgt_X_t[:, i]
                y_src_t = self.spatial_transform(src_t, inf_flow)
                y_tgt_t = self.spatial_transform(tgt_t, neg_inf_flow)

                recon_loss_tgt = (self.recon_loss(tgt, y_src) + self.recon_loss(tgt_t, y_src_t)) / 2
                recon_loss_tgt_l1 = (self.velocity_loss.recon_loss(tgt, y_src) + self.velocity_loss.recon_loss(tgt_t, y_src_t)) / 2
                recon_loss_src = (self.recon_loss(src, y_tgt) + self.recon_loss(src_t, y_tgt_t)) / 2
                recon_loss_src_l1 = (self.velocity_loss.recon_loss(src, y_tgt) + self.velocity_loss.recon_loss(src_t, y_tgt_t)) / 2

                kl_param_loss = self.velocity_loss.kl_loss(tgt, flow_param)
                smooth_loss_inf = self.velocity_loss.gradient_loss(inf_flow)
                smooth_loss_neg = self.velocity_loss.gradient_loss(neg_inf_flow)

                training_loss = training_loss + 0.01 * kl_param_loss + 0.5 * (recon_loss_tgt + recon_loss_src) + 0.01 * (recon_loss_tgt_l1 + recon_loss_src_l1) + self.alpha * (smooth_loss_inf + smooth_loss_neg)

        if self.lag_regular:
            # Lagrangian flow
            lag_flow = self.lag_flow(torch.stack(inf_flow_all, dim=2))
            
            # Warp the reference frame by the Lagrangian flow
            src_start_re = src_X[:, 0:1].repeat(1, t, 1, 1, 1)  # repeat the 1st frame to match other frames contained in a sequence
            src_start_re = rearrange(src_start_re, 'b t c h w  -> (b t) c h w')
            lag_y_src = self.spatial_transform(src_start_re.contiguous(), rearrange(lag_flow, 'b c t h w  -> (b t) c h w'))

            # Neg Lagrangian flow
            neg_lag_flow = self.lag_flow(torch.flip(torch.stack(neg_inf_flow_all, dim=2), dims=[2]))
            # Warp the reference frame by the Lagrangian flow
            src_end_re = tgt_X[:, -1:].repeat(1, t, 1, 1, 1)  # repeat the 1st frame to match other frames contained in a sequence
            src_end_re = rearrange(src_end_re, 'b t c h w  -> (b t) c h w')
            lag_neg_y_src = self.spatial_transform(src_end_re.contiguous(), rearrange(neg_lag_flow, 'b c t h w  -> (b t) c h w'))


        if train:
            if self.lag_regular:
                src_start_re_t = src_X_t[:, 0:1].repeat(1, t, 1, 1, 1)  # repeat the 1st frame to match other frames contained in a sequence
                src_start_re_t = rearrange(src_start_re_t, 'b t c h w  -> (b t) c h w')
                lag_y_src_t = self.spatial_transform(src_start_re_t.contiguous(), rearrange(lag_flow, 'b c t h w  -> (b t) c h w'))

                src_end_re_t = tgt_X_t[:, -1:].repeat(1, t, 1, 1, 1)  # repeat the 1st frame to match other frames contained in a sequence
                src_end_re_t = rearrange(src_end_re_t, 'b t c h w  -> (b t) c h w')
                lag_neg_y_src_t = self.spatial_transform(src_end_re_t.contiguous(), rearrange(neg_lag_flow, 'b c t h w  -> (b t) c h w'))

                lag_gradient_loss = self.velocity_loss.gradient_loss(lag_flow) + self.velocity_loss.gradient_loss(neg_lag_flow)
                recon_loss_lag = (self.recon_loss(rearrange(tgt_X, 'b t c h w  -> (b t) c h w'), lag_y_src) + 
                                  self.recon_loss(rearrange(tgt_X_t, 'b t c h w  -> (b t) c h w'), lag_y_src_t)) / 2
                recon_loss_neg_lag = (self.recon_loss(rearrange(torch.flip(src_X, dims=[1]), 'b t c h w  -> (b t) c h w'), lag_neg_y_src) + 
                                      self.recon_loss(rearrange(torch.flip(src_X_t, dims=[1]), 'b t c h w  -> (b t) c h w'), lag_neg_y_src_t))  / 2
                recon_loss_lag_l1 = (self.velocity_loss.recon_loss(rearrange(tgt_X, 'b t c h w  -> (b t) c h w'), lag_y_src) +
                                     self.velocity_loss.recon_loss(rearrange(tgt_X_t, 'b t c h w  -> (b t) c h w'), lag_y_src_t))  / 2
                recon_loss_neg_lag_l1 = (self.velocity_loss.recon_loss(rearrange(torch.flip(src_X, dims=[1]), 'b t c h w  -> (b t) c h w'), lag_neg_y_src) +
                                         self.velocity_loss.recon_loss(rearrange(torch.flip(src_X_t, dims=[1]), 'b t c h w  -> (b t) c h w'), lag_neg_y_src_t))  / 2

                training_loss = training_loss + lag_gradient_loss + self.beta * 0.5 * (recon_loss_lag + recon_loss_neg_lag) + 0.01 * (recon_loss_lag_l1 + recon_loss_neg_lag_l1)

            return (kl_param_loss, recon_loss_tgt, recon_loss_src, smooth_loss_inf, smooth_loss_neg, lag_gradient_loss, recon_loss_lag + recon_loss_neg_lag, recon_loss_lag_l1 + recon_loss_neg_lag_l1, training_loss), \
                    inf_flow_all, neg_inf_flow_all, lag_flow, rearrange(lag_y_src, '(b t) c h w -> b t c h w', b=b, t=t), inf_regsiter, neg_inf_regsiter
        
        return inf_flow_all, neg_inf_flow_all, lag_flow, neg_lag_flow, rearrange(lag_y_src, '(b t) c h w -> b t c h w', b=b, t=t), inf_regsiter, neg_inf_regsiter

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1)

def RViT_P8_L16_112(**kwargs):
    input_size = 256
    patch_size = 16
    num_layers = 2
    mlp_dim = 64
    length = 32

    return RViT(
                image_size = input_size,
                patch_size = patch_size,
                depth = num_layers,
                length = length,
                heads = 8,
                mlp_dim = mlp_dim,
                dropout = 0.1,
                emb_dropout = 0.1
            )


if __name__ == '__main__':
    import torch
    using_device = 'cuda:8'
    input_size = 256
    patch_size = 16
    num_layers = 2
    mlp_dim = 64
    length = 32

    v = RViT(image_size = input_size,
             patch_size = patch_size,
             depth = num_layers,
             length = length,
             heads = 8,
             mlp_dim = mlp_dim,
             dropout = 0.1,
             emb_dropout = 0.1
            ).cuda(using_device)

    vid = torch.randn(1, length + 1, 1, input_size, input_size).cuda(using_device)
    hidden = torch.zeros(1, (input_size//patch_size)**2, mlp_dim).cuda(using_device)

    from thop import profile, clever_format
    import time
    sum_ = 0
    for name, param in v.named_parameters():
        mul = 1
        if 'transformer' in name:
            for size_ in param.shape:
                mul *= size_
            sum_ += mul
            # print('%14s : %s, %1.f' % (name, param.shape, mul))
        # print('%s' % param)
    print('Params M:', sum_)

    torch.cuda.synchronize()
    start=time.time()
    macs, params = profile(v, inputs=(vid, hidden))
    torch.cuda.synchronize()
    end=time.time()
    print("time:", (end-start)/length)
    print("TFlops:", (macs*2 / 100000000000)/length)

    preds = v(vid, hidden)
    # print(preds[0])
