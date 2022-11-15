# SpecGAN: Large-factor Super-resolution of Remote Sensing Images
Yapeng Meng, Wenyuan Li, Sen Lei, Zhengxia Zou, and Zhenwei Shi*

[![IEEE](https://ieeexplore.ieee.org/assets/img/ieee_logo_white.svg)](https://ieeexplore.ieee.org/document/9950553)

The official PyTorch implementation of SpecGAN, Large-factor Super-resolution of Remote Sensing Images with Spectra-guided Generative Adversarial Networks, accepted by IEEE Transactions on Geoscience and Remote Sensing（TGRS).

## Abstract
Large-factor image super-resolution is a challenging task due to the high uncertainty and incompleteness of the missing details to be recovered. In remote sensing images, the sub-pixel spectral mixing and semantic ambiguity of ground objects make this task even more challenging. In this paper, we propose a novel method for large-factor super-resolution of remote sensing images named ``Spectra-guided Generative Adversarial Networks (SpecGAN)''. In response to the above problems, we explore whether introducing additional hyperspectral images to GAN as conditional input can be the key to solving the problems. Different from previous approaches that mainly focus on improving the feature representation of a single source input, we propose a dual branch network architecture to effectively fuse low-resolution RGB images and corresponding hyperspectral images, which fully exploit the rich hyperspectral information as conditional semantic guidance. Due to the spectral specificity of ground objects, the semantic accuracy of the generated images is guaranteed. To further improve the visual fidelity of the generated output, we also introduce the Latent Code Bank with rich visual priors under a generative adversarial training framework so that high-resolution, detailed, and realistic images can be progressively generated. Extensive experiments show the superiority of our method over the state-of-art image super-resolution methods in terms of both quantitative evaluation metrics and visual quality. Ablation experiments also suggest the necessity of adding spectral information and the effectiveness of our designed fusion module. To our best knowledge, we are the first to achieve up to 32x super-resolution of remote sensing images with high visual fidelity under the premise of accurate ground object semantics.

## Visualization results
![](imgs/teaser.jpg?20x15)

## Installation

SpecGAN builds on [MMEditing](https://github.com/open-mmlab/mmediting) and [MMCV](https://github.com/open-mmlab/mmcv). 
**We make some necessary modifications in order to load hyperspectral images (HSI).**

**Step 1.**
Clone our repository and create conda environment
```shell
conda env create -f environment.yml
```

**Step 2.**
Install `gdal`

If you use Windows and Python 3.7, just download [gdal](https://download.lfd.uci.edu/pythonlibs/archived/cp37/GDAL-3.4.2-cp37-cp37m-win_amd64.whl) wheel.
```shell
pip install GDAL-3.4.2-cp37-cp37m-win_amd64.whl
```

**Step 3.**
Install MMCV with [MIM](https://github.com/open-mmlab/mim).

```shell
pip install openmim
mim install mmcv-full==1.3.11
```
Please use our provide`SpecGAN/photometric.py` to replace `yourcondaroot/Anaconda3/envs/SpecGAN/Lib/site-packages/mmcv/image/photometric.py` to support muti-channel image processing.

**Step 4.**
Install our modified `mmediting`. 
```shell
pip install -e .
```

## Train
**Step 1.**

Edit config files at `configs/SpecGAN_32x_sr.py` or `configs/SpecGAN_16x_sr.py` to modify `line 87-112` to switch to your dataset path.

The data set should be place as [here](#pretrained-model-and-dataset).

**Step 2.**

x32 super-resolution:
```
python tools/train.py configs/SpecGAN_32x_sr.py
```
x16 super-resolution:
```
python tools/train.py configs/SpecGAN_16x_sr.py
```
It needs a really long training time, about 14 days on a GTX 1080Ti graphic card.
## Evaluate

x32 super-resolution:
```
python tools/test.py configs/SpecGAN_32x_sr.py work_dirs/xxx.pth --save-path xxx
```
x16 super-resolution:
```
python tools/test.py configs/SpecGAN_16x_sr.py work_dirs/xxx.pth --save-path xxx
```
## Pretrained model and dataset
Our pre-trained models can be download from [Google Drive](https://drive.google.com/drive/folders/1Tm4r5U1PKuvm9NV5QouFsrL7pR6Lrco_?usp=share_link) or [百度网盘](https://pan.baidu.com/s/1QQruAOTB3IL5Kuo87GVgAA?pwd=gep7).

The dataset should be arranged as follows:
```angular2html
--data_root_path
    --train
        --RGBHR
            --1.tif
            --2.tif
            --...
        --HSI
            --1.tif
            --2.tif
            --...
    --valid
        ...
    --test
        ...
```
Our paper uses dataset from the [IEEE data fusion contest 2019](https://dx.doi.org/10.21227/c6tm-vw12).

The HSI input image should be resized to the same size as low resolution RGB image.
We concat the two input as `[HSI;RGBLR]`, which has 48 channels and 3 channels, respectively.

## Citation
If our research is helpful to your research, please cite it as below.
```bibtex
@ARTICLE{9950553,
  author={Meng, Yapeng and Li, Wenyuan and Lei, Sen and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Large-factor Super-resolution of Remote Sensing Images with Spectra-guided Generative Adversarial Networks}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2022.3222360}}
```

And our code builds on [MMEditing](https://github.com/open-mmlab/mmediting), please cite as follows:

```bibtex
@misc{mmediting2022,
    title = {{MMEditing}: {OpenMMLab} Image and Video Editing Toolbox},
    author = {{MMEditing Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year = {2022}
}
```
