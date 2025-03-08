# Code Implementation of the LoShNet Model
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import time

from basicsr.archs.shift_scan_util import losh_ids_generate, losh_ids_scan, losh_ids_inverse, losh_shift_ids_generate
# from shift_scan_util import losh_ids_generate, losh_ids_scan, losh_ids_inverse, losh_shift_ids_generate
# try:
#     from .csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
# except:
#     from csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex

NEG_INF = -1000000

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].reshape(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30, input_resolution=(64,64)):
        super(CAB, self).__init__()
        self.is_light_sr = is_light_sr
        self.input_resolution = input_resolution
        self.num_feat = num_feat
        self.compress_ratio = compress_ratio

        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.is_light_sr:
            flops += H * W * self.num_feat * (self.num_feat // 6) * 9
            flops += H * W * (self.num_feat // 6)
            flops += H * W * (self.num_feat // 6) * self.num_feat * 9
        else:
            flops += H * W * self.num_feat * (self.num_feat // self.compress_ratio) * 9
            flops += H * W * (self.num_feat // self.compress_ratio)
            flops += H * W * (self.num_feat // self.compress_ratio) * self.num_feat * 9
        # print("flops in cab:", flops/1e6, self.compress_ratio)

        # flops_cab = H * W * self.num_feat * (self.num_feat // self.compress_ratio) * 9 * 2 + H * W * (self.num_feat // self.compress_ratio)
        # flops_mlp = self.num_feat * self.num_feat * H * W * 2 * 2
        # print("Ratio of Convblock/MLP in the CAB: %.2f"%(flops_cab/flops_mlp))

        # flops of ChannelAttention
        # flops of nn.AdaptiveAvgPool2d(1)
        flops += H * W * self.num_feat 
        # flops of nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0)
        flops += 1 * self.num_feat * (self.num_feat // self.compress_ratio)
        flops += 1 * (self.num_feat // self.compress_ratio)
        flops += 1 * (self.num_feat // self.compress_ratio) * self.num_feat
        flops += 1 * self.num_feat
        flops += H * W * self.num_feat
        # print("Ratio of flops_cab/flops in the CAB: %.2f, %d, %d"%(flops_cab/flops, flops_cab, flops))
        # 几乎没变，主要的计算量在conv
        return flops

class LoSS2Dv2(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=1,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            input_resolution=(64, 64),
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.input_resolution = input_resolution
        self.oact = True

        # self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj = Linear2d(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # TODO: verify whether use it
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        k_group=4
        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear2d(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # print(self.x_proj_weight.shape)
        # initialize in ["v2"]
        self.Ds = nn.Parameter(torch.ones((k_group * self.d_inner)))
        self.A_logs = nn.Parameter(torch.zeros((k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, self.d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, self.d_inner)))

        self.selective_scan = selective_scan_fn
        # self.selective_scan = SelectiveScanCore

        # self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_norm = nn.LayerNorm(self.d_inner)

    def forward_core(self, x: torch.Tensor, 
                     losh_ids,
                     x_proj_bias: torch.Tensor=None,
                     delta_softplus = True,
                     ssoflex=False
                     ):
        # print(x.shape) C=360
        B, C, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape

        L = H * W
        K = 4

        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        xs_scan_ids, xs_inverse_ids = losh_ids

        xs = losh_ids_scan(x, xs_scan_ids, bkdl=True) # (B, 4, -1, L)
        if True:
            x_dbl = F.conv1d(xs.reshape(B, -1, L), x_proj_weight.reshape(-1, D, 1), bias=(x_proj_bias.reshape(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
            dts = F.conv1d(dts.contiguous().reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.reshape(B, -1, L)
        dts = dts.reshape(B, -1, L).float() # (b, k * d, l)
        As = -torch.exp(A_logs.to(torch.float))
        Bs = Bs.reshape(B, K, N, L)
        Cs = Cs.reshape(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float) # [K*c]

        # def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        #     # return SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows=1, backnrows=1, ssoflex=False)
        #     # return self.selective_scan.forward(u, delta, A, B, C, D, delta_bias, delta_softplus)
        #     return self.selective_scan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus)

        # ys: torch.Tensor = self.selective_scan(
        #     xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        # )
        ys = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        ).reshape(B, K, -1, L)
        assert ys.dtype == torch.float

        # ys = ys.reshape(B, K, -1, L)
        # .view(B, K, -1, H, W)

        y1, y2, y3, y4 = losh_ids_inverse(ys, xs_inverse_ids, (B, C, H, W))

        return y1, y2, y3, y4

        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y


    def forward(self, x: torch.Tensor, losh_ids, **kwargs):

        B, C, H, W = x.shape
        # x = x.reshape(B, H, W, C)

        x = self.in_proj(x)
        # x, z = xz.chunk(2, dim=-1)

        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x, losh_ids)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        # y = self.out_act(y).reshape(B, C, H, W)
        y = self.out_act(y)
        # y = y * F.silu(z)self.out_act
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    def flops_forward_core(self, H, W):
        flops = 0
        # flops of x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) in Core
        flops += 4 * (H * W) * self.d_inner * (self.dt_rank + self.d_state * 2)
        # flops of dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dt_rank=12, d_inner=360
        flops += 4 * (H * W) * self.dt_rank * self.d_inner
        # print(flops/1e6, (4 * H * W) * (self.d_state * self.d_state * 2)/1e6)
        # 610.46784 M 8.388608 M

        # Flops of discretization
        flops += (4 * H * W) * (self.d_state * self.d_state * 2)

        # Flops of MambaIR selective_scan
        # # h' = Ah(t) + Bx(t)
        # flops += (4 * H * W) * (self.d_state * self.d_state + self.d_inner * self.d_state)
        # # y = Ch(t) + DBx(t)
        # flops += (4 * H * W) * (self.d_inner * self.d_inner + self.d_inner * self.d_state)
        # 640*360*36*90*16/1e9=11.94G 
        flops += 4 * 9 * H * W * self.d_inner * self.d_state
        # print(4 * 9 * H * W * self.d_inner * self.d_state/1e9)

        return flops
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # flop of in_proj
        flops += H * W * self.d_model * self.d_inner
        # flops of x = self.act(self.conv2d(x))
        flops += H * W * self.d_inner * 3 * 3 + H * W * self.d_inner
        # print(H, W, self.d_state, self.d_inner)
        flops += self.flops_forward_core(H, W)
        # 64 64 16 360
        # y = y1 + y2 + y3 + y4
        flops += 4 * H * W * self.d_inner
        # flops of y = self.out_act(y)
        flops += H * W * self.d_inner

        # flops of out = self.out_proj(y)
        flops += H * W * self.d_inner * self.d_model

        return flops


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            input_resolution= (64, 64),
            is_light_sr: bool = False,
            shift_size=0,
            **kwargs,
    ):
        super().__init__()
        # self.ln_1 = norm_layer(hidden_dim)
        self.ln_1 = LayerNorm2d(hidden_dim)
        self.self_attention = LoSS2Dv2(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, input_resolution=input_resolution, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(1, hidden_dim, 1, 1))
        self.conv_blk = CAB(hidden_dim,is_light_sr,input_resolution=input_resolution)
        # self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ln_2 = LayerNorm2d(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(1, hidden_dim, 1, 1))
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution

        self.shift_size = shift_size

    def forward(self, input, losh_ids, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        # B, C, H, W = input.shape
        # input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        input = input.reshape(B, C, *x_size)  # [B,C,H,W]

        # cyclic shift
        xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids = losh_ids
        if self.shift_size > 0:
            losh_ids = (xs_shift_scan_ids, xs_shift_inverse_ids)
        else:
            losh_ids = (xs_scan_ids, xs_inverse_ids)

        x = self.ln_1(input)

        # x = input*self.skip_scale + self.drop_path(self.self_attention(x, losh_ids))
        x = input*self.skip_scale.reshape(1, -1, 1, 1) + self.drop_path(self.self_attention(x, losh_ids))
        
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x))

        x = x.reshape(B, -1, C)
        return x
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # flops of norm1 self.ln_1 -> layer_norm1
        flops += self.hidden_dim * H * W
        # flops of SS2D
        flops += self.self_attention.flops()
        # flops of input * self.skip_scale and residual
        flops += self.hidden_dim * H * W * 2 
        # flops of norm2 self.ln_2 -> layer_norm2
        flops += self.hidden_dim * H * W 
        # flops of CAB
        flops += self.conv_blk.flops()
        # flops of input * self.skip_scale2 and residual
        flops += self.hidden_dim * H * W * 2 

        # flops_mlp = self.hidden_dim * self.hidden_dim * H * W * 4 * 2
        # print("flops of attn:: %.2f M, Per:%.2f"%(self.self_attention.flops()/1e6, self.self_attention.flops()/flops))
        # print("flops of conv_blk:: %.2f M, Per:%.2f"%(self.conv_blk.flops()/1e6, self.conv_blk.flops()/flops))
        # print("Ratio of Convblock/MLP:: %.2f"%(self.conv_blk.flops()/flops_mlp))
        # print(flops/1e6)

        return flops
    


class BasicLayer(nn.Module):
    """ The Basic LoShNet Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_light_sr=False,
                 scan_len=4
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,
                is_light_sr=is_light_sr,
                shift_size=0 if (i % 2 == 0) else scan_len // 2))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, losh_ids, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, losh_ids, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class LoSSV2ssm(nn.Module):
    r""" LoShNet Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.
           
           Local-Scan Shift-Path Network 

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=60,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state=2,
                 mlp_ratio=1.5,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffledirect',
                 resi_connection='1conv',
                 dynamic_ids=False,
                 scan_len=8,
                 batch_size=1,
                 **kwargs):

        super(LoSSV2ssm, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).reshape(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.num_out_ch = num_out_ch

        self.dynamic_ids = dynamic_ids
        self.scan_len = scan_len
        
        img_size_ids = to_2tuple(img_size)

        if not self.dynamic_ids:
            self._generate_ids((batch_size, embed_dim, img_size_ids[0], img_size_ids[1]))

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr = self.is_light_sr,
                scan_len=scan_len
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def _generate_ids(self, inp_shape):
        B,C,H,W = inp_shape

        xs_scan_ids, xs_inverse_ids = losh_ids_generate(inp_shape=(B, int(C*self.mlp_ratio), H, W), scan_len=self.scan_len)# [B,H,W,C]
        self.xs_scan_ids = xs_scan_ids
        self.xs_inverse_ids = xs_inverse_ids

        xs_shift_scan_ids, xs_shift_inverse_ids = losh_shift_ids_generate(inp_shape=(B, int(C*self.mlp_ratio), H, W), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        self.xs_shift_scan_ids = xs_shift_scan_ids
        self.xs_shift_inverse_ids = xs_shift_inverse_ids

    def forward_features(self, x):
        B,C,H,W = x.shape
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C
        x = self.pos_drop(x)

        # start = time.time()
        if self.dynamic_ids or (not self.training):
            xs_scan_ids, xs_inverse_ids = losh_ids_generate(inp_shape=(B, int(C*self.mlp_ratio), H, W), scan_len=self.scan_len)# [B,H,W,C]
            xs_shift_scan_ids, xs_shift_inverse_ids = losh_shift_ids_generate(inp_shape=(B, int(C*self.mlp_ratio), H, W), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        else:
            xs_scan_ids, xs_inverse_ids = self.xs_scan_ids, self.xs_inverse_ids
            xs_shift_scan_ids, xs_shift_inverse_ids = self.xs_shift_scan_ids, self.xs_shift_inverse_ids
        if torch.cuda.is_available():
            xs_scan_ids, xs_inverse_ids = xs_scan_ids.cuda(), xs_inverse_ids.cuda()
            xs_shift_scan_ids, xs_shift_inverse_ids = xs_shift_scan_ids.cuda(), xs_shift_inverse_ids.cuda()


        for layer in self.layers:
            x = layer(x, (xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids), x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x


    def flops_layers(self):
        flops = 0
        h, w = self.patches_resolution

        # flops of forward_features
        flops += self.patch_embed.flops()
        # print("self.patches_resolution:", self.patches_resolution)

        for layer in self.layers:
            flops += layer.flops()

        # flops of self.norm
        flops += h * w * self.embed_dim 

        # flops of self.patch_unembed
        flops += h * w * 9 * self.embed_dim * self.embed_dim

        # flops of self.conv_after_body
        flops += h * w * 9 * self.embed_dim * self.embed_dim

        # flops of Residual
        flops += h * w * self.embed_dim

        return flops

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        # x = self.conv_first(x)
        flops += h * w * 3 * self.embed_dim * 9

        if self.upsampler == 'pixelshuffle':
            # for classical SR

            # x = self.conv_after_body(self.forward_features(x)) + x
            flops += self.flops_layers()

            # x = self.conv_before_upsample(x)
            # nn.Conv2d(embed_dim, num_feat (=64), 3, 1, 1), nn.LeakyReLU(inplace=True))
            flops += h * w * 9 * self.embed_dim * 64
            flops += h * w * 64

            # self.upsample(x)
            if self.upscale == 2:
                flops += h * w * 9 * 64 * 4*64
            elif self.upscale == 3:
                flops += h * w * 9 * 64 * 9*64
            # x = self.conv_last()
            flops += h * w * 9 * 64 * 3

        elif self.upsampler == 'pixelshuffledirect':
            # x = self.conv_after_body(self.forward_features(x)) + x
            flops += self.flops_layers()

            # flops of UpsampleOneStep
            # self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
            flops += h * w * 9 * self.embed_dim * (self.upscale**2) * self.num_out_ch

        return flops


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr = False,
                 scan_len=4
                ):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr,
            scan_len=scan_len
            )

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, losh_ids, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, losh_ids, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).reshape(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def cal_test(net, inp):
    # net.eval()
    torch.cuda.synchronize()
    start = time.time()
    result = net(inp)
    torch.cuda.synchronize()
    end = time.time()
    print("network time:", end-start, " s")


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    torch.cuda.set_device(4)
    net = LoSSV2ssm(img_size=(64, 64), d_state=1, dynamic_ids=False, batch_size=1).cuda()
    print('FLOPS calculated by Ours: %.2f G'%(net.flops()/1e9))
    # inp = torch.tensor
    from thop import profile
    inp = torch.randn(1, 3, 64, 64).cuda()
    # 64, 64 640, 360
    # cal_test(net, inp)

    # flops, params, ret_dict = profile(net, inputs=(inp,), verbose=True, ret_layer_info=True)
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')

    print(get_parameter_number(net))
