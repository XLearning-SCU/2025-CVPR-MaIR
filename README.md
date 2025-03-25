# [CVPR 2025] MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration

[[Paper](https://arxiv.org/abs/2412.20066)]

Boyun Li, Haiyu Zhao, Wenxin Wang, Peng Hu, Yuanbiao Gou*, Xi Peng*

> **Abstract:**  Recent advancements in Mamba have shown promising results in image restoration. These methods typically flatten 2D images into multiple distinct 1D sequences along rows and columns, process each sequence independently using selective scan operation, and recombine them to form the outputs. However, such a paradigm overlooks two vital aspects: i) the local relationships and spatial continuity inherent in natural images, and ii) the discrepancies among sequences unfolded through totally different ways. To overcome the drawbacks, we explore two problems in Mamba-based restoration methods: i) how to design a scanning strategy preserving both locality and continuity while facilitating restoration, and ii) how to aggregabuildMaIR_SmallbuildMaIR_Smallte the distinct sequences unfolded in totally different ways. To address these problems, we propose a novel Mamba-based Image Restoration model (MaIR), which consists of Nested S-shaped Scanning strategy (NSS) and Sequence Shuffle Attention block (SSA). Specifically, NSS preserves locality and continuity of the input images through the stripe-based scanning region and the S-shaped scanning path, respectively. SSA aggregates sequences through calculating attention weights within the corresponding channels of different sequences. Thanks to NSS and SSA, MaIR surpasses 40 baselines across 14 challenging datasets, achieving state-of-the-art performance on the tasks of image super-resolution, denoising, deblurring and dehazing.

There are the codes of MaIR. I will reformulate codes and options very very soon. (within 1-2 weeks).

CKPT can be referred at [here](https://drive.google.com/drive/folders/1YYmIVTyynLg-Kfu-mviq24WdVkJu-S3M?usp=sharing).

## Guidance for Training and Testing

### Training and Testing Commands on Super-Resolution

```bash
# Training commands for x2, x3, x4 Classic SR (~1 week for x2 SR)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_SR_x2.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_SR_x3.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_SR_x4.yml --launcher pytorch
# Training commands for small version of x2, x3, x4 lightSR (~1.3M) (~3 days)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_S_lightSR_x2.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_S_lightSR_x3.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_S_lightSR_x4.yml --launcher pytorch
# Training commands for tiny version of x2, x3, x4 lightSR (<900K) (~3 days)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_T_lightSR_x2.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_T_lightSR_x3.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_T_lightSR_x4.yml --launcher pytorch

# Testing commands for x2, x3, x4 Classic SR
python basicsr/test.py -opt options/test/test_MaIR_SR_x2.yml
python basicsr/test.py -opt options/test/test_MaIR_SR_x3.yml
python basicsr/test.py -opt options/test/test_MaIR_SR_x4.yml
# Testing commands for small version of x2, x3, x4 lightSR (~1.3M)
python basicsr/test.py -opt options/test/test_MaIR_S_lightSR_x2.yml
python basicsr/test.py -opt options/test/test_MaIR_S_lightSR_x3.yml
python basicsr/test.py -opt options/test/test_MaIR_S_lightSR_x4.yml
# Testing commands for tiny version of x2, x3, x4 lightSR (<900K)
python basicsr/test.py -opt options/test/test_MaIR_T_lightSR_x2.yml
python basicsr/test.py -opt options/test/test_MaIR_T_lightSR_x3.yml
python basicsr/test.py -opt options/test/test_MaIR_T_lightSR_x4.yml
```

### Training and Testing Commands on Color Image Denoising

```bash
# Training commands for Color Denoising with sigma=15  (~2 weeks)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_CDN_s15.yml --launcher pytorch
# Training commands for Color Denoising with sigma=25
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_CDN_s25.yml --launcher pytorch
# Training commands for Color Denoising with sigma=50
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_CDN_s50.yml --launcher pytorch
# Training commands for Real Denoising (1-2 weeks)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1268 realDenoising/basicsr/trainF.py -opt realDenoising/options/train/train_MaIR_RealDN.yml --launcher pytorch

# Testing commands for Color Denoising with sigma=15
python basicsr/test.py -opt options/test/test_MaIR_CDN_s15.yml
# Testing commands for Color Denoising with sigma=25
python basicsr/test.py -opt options/test/test_MaIR_CDN_s25.yml
# Testing commands for Color Denoising with sigma=50
python basicsr/test.py -opt options/test/test_MaIR_CDN_s50.yml
# Testing commands for Real Denoising
python basicsr/test.py -opt options/test/test_MaIR_RealDN.yml
```


### Training and Testing Commands on Motion Debluring

```bash
# Training commands for Motion Debluring (1-2 weeks)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1268 realDenoising/basicsr/trainF.py -opt realDenoising/options/train/train_MaIR_MotionDeblur.yml --launcher pytorch

# Testing commands for Motion Debluring
python realDenoising/basicsr/test.py -opt realDenoising/options/test/test_MaIR_MotionDeblur.yml
```

### Training and Testing Commands on Image Dehazing

```bash
# Training commands for Image Dehazing (~3 days)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=8 --master_port=1268 realDenoising/basicsr/trainF.py -opt realDenoising/options/train/train_MaIR_ITS.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=8 --master_port=1268 realDenoising/basicsr/trainF.py -opt realDenoising/options/train/train_MaIR_OTS.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=8 --master_port=1268 realDenoising/basicsr/trainF.py -opt realDenoising/options/train/train_MaIR_R6K.yml --launcher pytorch

# Testing commands for Image Dehazing
python realDenoising/basicsr/test.py -opt realDenoising/options/test/test_MaIR_ITS.yml
python realDenoising/basicsr/test.py -opt realDenoising/options/test/test_MaIR_OTS.yml
python realDenoising/basicsr/test.py -opt realDenoising/options/test/test_MaIR_R6K.yml
```


Cautions: torchrun is only available for pytorch>=1.9.0. If you do not want to use torchrun for training, you can replace it with `python -m torch.distributed.launch` for training.

For MaIR+: You could change the model_type in test options to MaIRPlusModel for MaIR+.

## TODO

* [X] Update codes of networks
* [X] Upload checkpoint
* [X] Update the codes of calculating flops and params through fvcore
* [X] update options for SR
* [X] update options for lightSR
* [X] update options for denoising
* [X] update options for realDN
* [X] update options for deblurring
* [X] update options for dehazing
* [X] Update MaIR+
* [x] update training and testing commands
* [X] update unet-version
* [ ] Improve codes of calculating FLOPs and Params.
* [ ] update readme
* [ ] update environments
* [ ] update homepage

## Citations

If our work is useful for your research, please consider citing:

```
@inproceedings{MaIR,
  title={MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration},
  author={Li, Boyun and Zhao, Haiyu and Wang, Wenxin and Hu, Peng and Gou, Yuanbiao and Peng, Xi},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2025},
  address = {Nashville, TN},
  month = jun
}
```

## Acknowledgement

This code is based on [MambaIR](https://github.com/csguoh/MambaIR/) and [VMamba](https://github.com/MzeroMiko/VMamba). Many thanks for their awesome work.

## Contact

If you have any questions, feel free to contact me at liboyun.gm@gmail.com
