# The Code Implementatio of MambaIR model for Real Image Denoising task
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange
import math
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial
import sys
sys.path.append('/xlearning/boyun/codes/MaIR/realDenoising')
from shift_scanf_util import mair_ids_generate, mair_ids_scan, mair_ids_inverse, mair_shift_ids_generate

NEG_INF = -1000000


class ShuffleAttn(nn.Module):
    def __init__(self, in_features, out_features, group=4, input_resolution=(64,64)):
        super().__init__()
        self.group = group
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def channel_shuffle(self, x):
        # batchsize, num_channels, height, width = x.data.size()
        batchsize, num_channels, height, width = x.shape
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x
    
    def channel_rearrange(self,x):
        # batchsize, num_channels, height, width = x.data.size()
        batchsize, num_channels, height, width = x.shape
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        x = x.reshape(batchsize, self.group, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, x):
        x = self.channel_shuffle(x)
        x = self.gating(x)
        x = self.channel_rearrange(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., input_resolution=(64,64)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.input_resolution = input_resolution
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        flops += 2 * H * W * self.in_features * self.hidden_features
        flops += H * W * self.hidden_features

        return flops


class LoSh2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            ssm_ratio=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
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
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(self.ssm_ratio * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.input_resolution = input_resolution

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
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

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # print(self.x_proj_weight.shape)

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.gating = ShuffleAttn(in_features=self.d_inner*4, out_features=self.d_inner*4, group=self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, 
                     losh_ids,
                     x_proj_bias: torch.Tensor=None,
                     ):
        # print(x.shape) C=360
        B, C, H, W = x.shape
        L = H * W
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        K=4

        xs_scan_ids, xs_inverse_ids = losh_ids

        xs = mair_ids_scan(x, xs_scan_ids)

        x_dbl = F.conv1d(xs.reshape(B, -1, L), self.x_proj_weight.reshape(-1, D, 1), bias=(x_proj_bias.reshape(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), self.dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # [360]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return mair_ids_inverse(out_y, xs_inverse_ids, shape=(B, -1, H, W)) #B, C, L

    def forward(self, x: torch.Tensor, losh_ids, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x, losh_ids)
        assert y.dtype == torch.float32
        y = y * self.gating(y)
        y1, y2, y3, y4 = torch.chunk(y, 4, dim=1)
        y = y1 + y2 + y3 + y4
        y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)
        y = y * F.silu(z)
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
        flops += H * W * self.d_model * self.d_inner * 2
        # flops of x = self.act(self.conv2d(x))
        flops += H * W * self.d_inner * 3 * 3 + H * W * self.d_inner
        # print(H, W, self.d_state, self.d_inner)
        flops += self.flops_forward_core(H, W)
        # 64 64 16 360
        # y = y1 + y2 + y3 + y4
        flops += 4 * H * W * self.d_inner
        # flops of y = self.out_norm(y)
        flops += H * W * self.d_inner
        # flops of y = y * F.silu(z)
        flops += 2 * H * W * self.d_inner

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
            ssm_ratio: float = 2.,
            input_resolution= (64, 64),
            is_light_sr: bool = False,
            shift_size=0,
            mlp_ratio=1.5,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = LoSh2D(d_model=hidden_dim, d_state=d_state,ssm_ratio=ssm_ratio,dropout=attn_drop_rate, input_resolution=input_resolution, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,input_resolution=input_resolution)
        
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution

        self.shift_size = shift_size

    def forward(self, input, losh_ids, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]

        # cyclic shift
        xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids = losh_ids
        if self.shift_size > 0:
            losh_ids = (xs_shift_scan_ids, xs_shift_inverse_ids)
        else:
            losh_ids = (xs_scan_ids, xs_inverse_ids)

        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x, losh_ids))
        
        x = x*self.skip_scale2 + self.mlp(self.ln_2(x))

        x = x.view(B, -1, C).contiguous()
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
        flops += self.mlp.flops()
        # flops of input * self.skip_scale2 and residual
        flops += self.hidden_dim * H * W * 2 

        return flops
    

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class MaIRUNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 ssm_ratio=1.5,
                 num_refinement_blocks=4,
                 drop_path_rate=0.,
                 bias=False,
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 flp_ratio=2,
                 mlp_ratio=2,
                 dynamic_ids=False,
                 img_size=64,
                 scan_len=8,
                 batch_size=1,
                 ):

        super(MaIRUNet, self).__init__()
        self.ssm_ratio = ssm_ratio
        self.dynamic_ids = dynamic_ids
        self.scan_len = scan_len
        img_size_ids = to_2tuple(img_size)
        self.trainig_img_size = img_size
        if not self.dynamic_ids:
            self._generate_ids((batch_size, dim, img_size_ids[0], img_size_ids[1]))

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        base_d_state = 4
        self.encoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                ssm_ratio=self.ssm_ratio,
                d_state=base_d_state,
                mlp_ratio=flp_ratio,
            )
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                ssm_ratio=self.ssm_ratio,
                d_state=int(base_d_state * 2 ** 1),
                mlp_ratio=mlp_ratio,
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                ssm_ratio=self.ssm_ratio,
                d_state=int(base_d_state * 2 ** 2),
                mlp_ratio=mlp_ratio,
            )
            for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 3),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                ssm_ratio=self.ssm_ratio,
                d_state=int(base_d_state * 2 ** 3),
                # d_state=int(base_d_state * 2 ** 3),
                mlp_ratio=mlp_ratio,
            )
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                ssm_ratio=self.ssm_ratio,
                d_state=int(base_d_state * 2 ** 2),
                mlp_ratio=mlp_ratio,
            )
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                ssm_ratio=self.ssm_ratio,
                d_state=int(base_d_state * 2 ** 1),
                mlp_ratio=mlp_ratio,
            )
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                ssm_ratio=self.ssm_ratio,
                d_state=int(base_d_state * 2 ** 1),
                mlp_ratio=mlp_ratio,
                # d_state=int(base_d_state),
            )
            for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                ssm_ratio=self.ssm_ratio,
                d_state=int(base_d_state * 2 ** 1),
                mlp_ratio=mlp_ratio,
                # d_state=int(base_d_state),
            )
            for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def _generate_ids(self, inp_shape):
        B,C,H,W = inp_shape

        xs_scan_ids_l1, xs_inverse_ids_l1 = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)# [B,H,W,C]
        xs_scan_ids_l2, xs_inverse_ids_l2 = mair_ids_generate(inp_shape=(1, 1, H//2, W//2), scan_len=self.scan_len)# [B,H,W,C]
        xs_scan_ids_l3, xs_inverse_ids_l3 = mair_ids_generate(inp_shape=(1, 1, H//4, W//4), scan_len=self.scan_len)# [B,H,W,C]
        xs_scan_ids_lat, xs_inverse_ids_lat = mair_ids_generate(inp_shape=(1, 1, H//8, W//8), scan_len=self.scan_len)# [B,H,W,C]

        xs_shift_scan_ids_l1, xs_shift_inverse_ids_l1 = mair_shift_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        xs_shift_scan_ids_l2, xs_shift_inverse_ids_l2 = mair_shift_ids_generate(inp_shape=(1, 1, H//2, W//2), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        xs_shift_scan_ids_l3, xs_shift_inverse_ids_l3 = mair_shift_ids_generate(inp_shape=(1, 1, H//4, W//4), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        xs_shift_scan_ids_lat, xs_shift_inverse_ids_lat = mair_shift_ids_generate(inp_shape=(1, 1, H//8, W//8), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]

        if torch.cuda.is_available():
            self.xs_scan_ids_l1 = xs_scan_ids_l1.cuda()
            self.xs_scan_ids_l2 = xs_scan_ids_l2.cuda()
            self.xs_scan_ids_l3 = xs_scan_ids_l3.cuda()
            self.xs_scan_ids_lat = xs_scan_ids_lat.cuda()
            self.xs_inverse_ids_l1 = xs_inverse_ids_l1.cuda()
            self.xs_inverse_ids_l2 = xs_inverse_ids_l2.cuda()
            self.xs_inverse_ids_l3 = xs_inverse_ids_l3.cuda()
            self.xs_inverse_ids_lat = xs_inverse_ids_lat.cuda()

            self.xs_shift_scan_ids_l1 = xs_shift_scan_ids_l1.cuda()
            self.xs_shift_scan_ids_l2 = xs_shift_scan_ids_l2.cuda()
            self.xs_shift_scan_ids_l3 = xs_shift_scan_ids_l3.cuda()
            self.xs_shift_scan_ids_lat = xs_shift_scan_ids_lat.cuda()
            self.xs_shift_inverse_ids_l1 = xs_shift_inverse_ids_l1.cuda()
            self.xs_shift_inverse_ids_l2 = xs_shift_inverse_ids_l2.cuda()
            self.xs_shift_inverse_ids_l3 = xs_shift_inverse_ids_l3.cuda()
            self.xs_shift_inverse_ids_lat = xs_shift_inverse_ids_lat.cuda()
        else:
            self.xs_scan_ids_l1 = xs_scan_ids_l1
            self.xs_scan_ids_l2 = xs_scan_ids_l2
            self.xs_scan_ids_l3 = xs_scan_ids_l3
            self.xs_scan_ids_lat = xs_scan_ids_lat
            self.xs_inverse_ids_l1 = xs_inverse_ids_l1
            self.xs_inverse_ids_l2 = xs_inverse_ids_l2
            self.xs_inverse_ids_l3 = xs_inverse_ids_l3
            self.xs_inverse_ids_lat = xs_inverse_ids_lat

            self.xs_shift_scan_ids_l1 = xs_shift_scan_ids_l1
            self.xs_shift_scan_ids_l2 = xs_shift_scan_ids_l2
            self.xs_shift_scan_ids_l3 = xs_shift_scan_ids_l3
            self.xs_shift_scan_ids_lat = xs_shift_scan_ids_lat
            self.xs_shift_inverse_ids_l1 = xs_shift_inverse_ids_l1
            self.xs_shift_inverse_ids_l2 = xs_shift_inverse_ids_l2
            self.xs_shift_inverse_ids_l3 = xs_shift_inverse_ids_l3
            self.xs_shift_inverse_ids_lat = xs_shift_inverse_ids_lat

        del xs_scan_ids_l1, xs_inverse_ids_l1, xs_scan_ids_l2, xs_inverse_ids_l2, xs_scan_ids_l3, xs_inverse_ids_l3, xs_scan_ids_lat, xs_inverse_ids_lat
        del xs_shift_scan_ids_l1, xs_shift_inverse_ids_l1, xs_shift_scan_ids_l2, xs_shift_inverse_ids_l2, xs_shift_scan_ids_l3, xs_shift_inverse_ids_l3, xs_shift_scan_ids_lat, xs_shift_inverse_ids_lat

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape
        # x_size = (H, W)
        # start = time.time()
        if self.training and (self.trainig_img_size != H):
            self._generate_ids((B, C, H, W))
            self.trainig_img_size = H

            ids_l1 = (self.xs_scan_ids_l1, self.xs_inverse_ids_l1, self.xs_shift_scan_ids_l1, self.xs_shift_inverse_ids_l1)
            ids_l2 = (self.xs_scan_ids_l2, self.xs_inverse_ids_l2, self.xs_shift_scan_ids_l2, self.xs_shift_inverse_ids_l2)
            ids_l3 = (self.xs_scan_ids_l3, self.xs_inverse_ids_l3, self.xs_shift_scan_ids_l3, self.xs_shift_inverse_ids_l3)
            ids_lat = (self.xs_scan_ids_lat, self.xs_inverse_ids_lat, self.xs_shift_scan_ids_lat, self.xs_shift_inverse_ids_lat)

        elif self.dynamic_ids or (not self.training):
            xs_scan_ids_l1, xs_inverse_ids_l1 = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)# [B,H,W,C]
            xs_scan_ids_l2, xs_inverse_ids_l2 = mair_ids_generate(inp_shape=(1, 1, H//2, W//2), scan_len=self.scan_len)# [B,H,W,C]
            xs_scan_ids_l3, xs_inverse_ids_l3 = mair_ids_generate(inp_shape=(1, 1, H//4, W//4), scan_len=self.scan_len)# [B,H,W,C]
            xs_scan_ids_lat, xs_inverse_ids_lat = mair_ids_generate(inp_shape=(1, 1, H//8, W//8), scan_len=self.scan_len)# [B,H,W,C]

            xs_shift_scan_ids_l1, xs_shift_inverse_ids_l1 = mair_shift_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
            xs_shift_scan_ids_l2, xs_shift_inverse_ids_l2 = mair_shift_ids_generate(inp_shape=(1, 1, H//2, W//2), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
            xs_shift_scan_ids_l3, xs_shift_inverse_ids_l3 = mair_shift_ids_generate(inp_shape=(1, 1, H//4, W//4), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
            xs_shift_scan_ids_lat, xs_shift_inverse_ids_lat = mair_shift_ids_generate(inp_shape=(1, 1, H//8, W//8), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
            if torch.cuda.is_available():
                ids_l1 = (xs_scan_ids_l1.cuda(), xs_inverse_ids_l1.cuda(), xs_shift_scan_ids_l1.cuda(), xs_shift_inverse_ids_l1.cuda())
                ids_l2 = (xs_scan_ids_l2.cuda(), xs_inverse_ids_l2.cuda(), xs_shift_scan_ids_l2.cuda(), xs_shift_inverse_ids_l2.cuda())
                ids_l3 = (xs_scan_ids_l3.cuda(), xs_inverse_ids_l3.cuda(), xs_shift_scan_ids_l3.cuda(), xs_shift_inverse_ids_l3.cuda())
                ids_lat = (xs_scan_ids_lat.cuda(), xs_inverse_ids_lat.cuda(), xs_shift_scan_ids_lat.cuda(), xs_shift_inverse_ids_lat.cuda())
            del xs_scan_ids_l1, xs_inverse_ids_l1, xs_scan_ids_l2, xs_inverse_ids_l2, xs_scan_ids_l3, xs_inverse_ids_l3, xs_scan_ids_lat, xs_inverse_ids_lat
            del xs_shift_scan_ids_l1, xs_shift_inverse_ids_l1, xs_shift_scan_ids_l2, xs_shift_inverse_ids_l2, xs_shift_scan_ids_l3, xs_shift_inverse_ids_l3, xs_shift_scan_ids_lat, xs_shift_inverse_ids_lat
        else:
                ids_l1 = (self.xs_scan_ids_l1, self.xs_inverse_ids_l1, self.xs_shift_scan_ids_l1, self.xs_shift_inverse_ids_l1)
                ids_l2 = (self.xs_scan_ids_l2, self.xs_inverse_ids_l2, self.xs_shift_scan_ids_l2, self.xs_shift_inverse_ids_l2)
                ids_l3 = (self.xs_scan_ids_l3, self.xs_inverse_ids_l3, self.xs_shift_scan_ids_l3, self.xs_shift_inverse_ids_l3)
                ids_lat = (self.xs_scan_ids_lat, self.xs_inverse_ids_lat, self.xs_shift_scan_ids_lat, self.xs_shift_inverse_ids_lat)

        inp_enc_level1 = self.patch_embed(inp_img)  # b,hw,c
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, ids_l1, [H, W])
            # def forward(self, input, losh_ids, x_size):
            # x = layer(x, (xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids), x_size)

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, ids_l2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, ids_l3, [H // 4, W // 4])

        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
        latent = inp_enc_level4
        for layer in self.latent:
            latent = layer(latent, ids_lat, [H // 8, W // 8])

        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, ids_l3, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, ids_l2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, ids_l1, [H, W])

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, ids_l1, [H, W])

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num/1e6, 'Trainable': trainable_num/1e6}

if __name__ == '__main__':
    torch.cuda.set_device(0)
    # Dehazing
    model = MaIRUNet(
        inp_channels=3,
        out_channels=3,
        dim=24,
        num_blocks=[2, 2, 3, 4],
        num_refinement_blocks=2,
        ssm_ratio=1.2,
        flp_ratio=2.0,
        mlp_ratio=2.0,
        bias=False,
        dual_pixel_task=False,
        img_size=128,
        scan_len=8,
    ).cuda()

    # Deblurring
    # model = MaIRUNet(
    #     inp_channels=3,
    #     out_channels=3,
    #     dim=48,
    #     num_blocks=[4, 6, 6, 8],
    #     num_refinement_blocks=4,
    #     ssm_ratio=2.0,
    #     flp_ratio=4.0,
    #     mlp_ratio=1.5,
    #     bias=False,
    #     dual_pixel_task=False,
    #     img_size=128,
    #     scan_len=4,
    # ).cuda()

    height = 256
    width = 256
    x = torch.randn((1, 3, height, width)).cuda()
    print(get_parameter_number(model))

    memory_usage = {}
    # 定义前向 hook
    def forward_hook(module, input, output):
        # memory_usage[module] = torch.cuda.memory_allocated()
        for name, mod in model.named_modules():
            if mod is module:
                memory_usage[name] = torch.cuda.memory_allocated()
                break

    # 注册 hook
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(forward_hook)
        hooks.append(hook)
    
    # 执行前向传播
    output = model(x)
    print(output.shape)

    # 获取最大显存分配量
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9  # 转换为GB
    print(f"最大显存分配量: {max_memory_allocated} GB")
    # 获取最大显存预留量
    max_memory_reserved = torch.cuda.max_memory_reserved() / 1e9  # 转换为GB
    print(f"最大显存预留量: {max_memory_reserved} GB")
