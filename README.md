# [CVPR 2025] MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration

[[Paper](https://arxiv.org/abs/2412.20066)]

Boyun Li, Haiyu Zhao, Wenxin Wang, Peng Hu, Yuanbiao Gou*, Xi Peng*

> **Abstract:**  Recent advancements in Mamba have shown promising results in image restoration. These methods typically flatten 2D images into multiple distinct 1D sequences along rows and columns, process each sequence independently using selective scan operation, and recombine them to form the outputs. However, such a paradigm overlooks two vital aspects: i) the local relationships and spatial continuity inherent in natural images, and ii) the discrepancies among sequences unfolded through totally different ways. To overcome the drawbacks, we explore two problems in Mamba-based restoration methods: i) how to design a scanning strategy preserving both locality and continuity while facilitating restoration, and ii) how to aggregabuildMaIR_SmallbuildMaIR_Smallte the distinct sequences unfolded in totally different ways. To address these problems, we propose a novel Mamba-based Image Restoration model (MaIR), which consists of Nested S-shaped Scanning strategy (NSS) and Sequence Shuffle Attention block (SSA). Specifically, NSS preserves locality and continuity of the input images through the stripe-based scanning region and the S-shaped scanning path, respectively. SSA aggregates sequences through calculating attention weights within the corresponding channels of different sequences. Thanks to NSS and SSA, MaIR surpasses 40 baselines across 14 challenging datasets, achieving state-of-the-art performance on the tasks of image super-resolution, denoising, deblurring and dehazing.

There are the codes of MaIR. I will reformulate codes and options very very soon. (within 1-2 weeks).

CKPT can be referred at [here](https://drive.google.com/drive/folders/1YYmIVTyynLg-Kfu-mviq24WdVkJu-S3M?usp=sharing).

## TODO

* [X] Update codes of networks
* [X] Upload checkpoint
* [X] Update the codes of calculating flops and params through fvcore
* [X] update options for SR
* [X] update options for lightSR
* [X] update options for denoising
* [ ] update options for deblurring
* [ ] update options for dehazing
* [ ] update training and testing commands
* [ ] update unet-version
* [ ] update readme
* [ ] update environments
* [ ] update homepage

# Citations

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
