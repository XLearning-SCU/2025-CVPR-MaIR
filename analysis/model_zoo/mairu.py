from realDenoising.basicsr.models.archs.mairunet_arch import MaIRUNet

def buildMaIRU():
    return MaIRUNet(
        inp_channels=3,
        out_channels=3,
        dim=24,
        num_blocks=[2, 2, 3, 4],
        num_refinement_blocks=2,
        ssm_ratio=1.2,
        fmlp_ratio=2.0,
        mlp_ratio=2.0,
        bias=False,
        dual_pixel_task=False,
        img_size=256
        )

def buildMaIRU_motiondeblur():
    return MaIRUNet(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        ssm_ratio=2.0,
        fmlp_ratio=4.0,
        mlp_ratio=1.5,
        bias=False,
        dual_pixel_task=False,
        img_size=128
        )
