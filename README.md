# SpecGAN: Large-factor Super-resolution of Remote Sensing Images
Yapeng Meng, Wenyuan Li, Sen Lei, Zhengxia Zou, and Zhenwei Shi*

The official PyTorch implementation of SpecGAN, Large-factor Super-resolution of Remote Sensing Images with Spectra-guided Generative Adversarial Networks, accepted by IEEE Transactions on Geoscience and Remote Sensing（TGRS).

## Abstract
Large-factor image super-resolution is a challenging task due to the high uncertainty and incompleteness of the missing details to be recovered. In remote sensing images, the sub-pixel spectral mixing and semantic ambiguity of ground objects make this task even more challenging. In this paper, we propose a novel method for large-factor super-resolution of remote sensing images named ``Spectra-guided Generative Adversarial Networks (SpecGAN)''. In response to the above problems, we explore whether introducing additional hyperspectral images to GAN as conditional input can be the key to solving the problems. Different from previous approaches that mainly focus on improving the feature representation of a single source input, we propose a dual branch network architecture to effectively fuse low-resolution RGB images and corresponding hyperspectral images, which fully exploit the rich hyperspectral information as conditional semantic guidance. Due to the spectral specificity of ground objects, the semantic accuracy of the generated images is guaranteed. To further improve the visual fidelity of the generated output, we also introduce the Latent Code Bank with rich visual priors under a generative adversarial training framework so that high-resolution, detailed, and realistic images can be progressively generated. Extensive experiments show the superiority of our method over the state-of-art image super-resolution methods in terms of both quantitative evaluation metrics and visual quality. Ablation experiments also suggest the necessity of adding spectral information and the effectiveness of our designed fusion module. To our best knowledge, we are the first to achieve up to 32x super-resolution of remote sensing images with high visual fidelity under the premise of accurate ground object semantics.

## Visualization results
![](imgs/teaser.jpg?20x15)

## Installation

SpecGAN builds on [MMEditing](https://github.com/open-mmlab/mmediting) and [MMCV](https://github.com/open-mmlab/mmcv). We make some necessary modifications in order to load hyperspectral images (HSI).

**Step 1.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

**Step 2.**
Install MMCV with [MIM](https://github.com/open-mmlab/mim).

```shell
pip3 install openmim
mim install mmcv-full
```

**Step 3.**
Install gdal (used to load muti-channal HSI images `.tif`)
```
conda install gdal
```

## Train
**Step 1.**
Edit config files at `configs/SpecGAN`.
**Step 2.**
```
python tools/train.py configs/SpecGAN/SpecGANx32.py
```
## Evaluate
```
python tools/test.py configs/SpecGAN/SpecGANx32.py work_dirs/xxx.pth --save-path xxx
```
## Pretrained model and dataset

## Citation

And our code builds on [MMEditing](https://github.com/open-mmlab/mmediting), please cite as follows:

```bibtex
@misc{mmediting2022,
    title = {{MMEditing}: {OpenMMLab} Image and Video Editing Toolbox},
    author = {{MMEditing Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year = {2022}
}
```
