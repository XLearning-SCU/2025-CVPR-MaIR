import torch.nn as nn
from basicsr.archs.mair_arch import MaIR

def buildMaIR_Small(upscale=2):
    return MaIR(img_size=64,
                   patch_size=1,
                   in_chans=3,
                   embed_dim=60,
                   depths=(6, 6, 6, 6),
                   mlp_ratio=1.6,
                   ssm_ratio=1.4,
                   drop_rate=0.,
                   norm_layer=nn.LayerNorm,
                   patch_norm=True,
                   use_checkpoint=False,
                   upscale=upscale,
                   img_range=1.,
                   upsampler='pixelshuffledirect',
                   resi_connection='1conv')

def buildMaIR_Tiny(upscale=2):
    return MaIR(img_size=64,
                   patch_size=1,
                   in_chans=3,
                   embed_dim=60,
                   depths=(6, 6, 6, 6),
                   mlp_ratio=1.6,
                   ssm_ratio=1.1,
                   d_state=1,
                   drop_rate=0.,
                   norm_layer=nn.LayerNorm,
                   patch_norm=True,
                   use_checkpoint=False,
                   upscale=upscale,
                   img_range=1.,
                   upsampler='pixelshuffledirect',
                   resi_connection='1conv')


def buildMaIR(upscale=2):
    return MaIR(img_size=64,
                   patch_size=1,
                   in_chans=3,
                   embed_dim=180,
                   depths=(6, 6, 6, 6, 6, 6),
                   mlp_ratio=2.,
                   drop_rate=0.,
                   norm_layer=nn.LayerNorm,
                   patch_norm=True,
                   use_checkpoint=False,
                   upscale=upscale,
                   img_range=1.,
                   upsampler='pixelshuffle',
                   resi_connection='1conv')

def buildMaIR_SR(upscale=2):
    return MaIR(img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=180,
                 depths=(6, 6, 6, 6, 6, 6),
                 drop_rate=0.,
                 d_state=16,
                 ssm_ratio=2.0,
                 mlp_ratio=2.5,

                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=upscale,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 dynamic_ids=False,
                 scan_len=4,
                 batch_size=1,
                )
