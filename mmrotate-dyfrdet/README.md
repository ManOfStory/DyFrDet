
## This repository is the official implementation of "DecoupleNet: A Lightweight Backbone Network with Efficient Feature Decoupling for Remote Sensing Visual Tasks".
> [**DecoupleNet: A Lightweight Backbone Network with Efficient Feature Decoupling for
Remote Sensing Visual Tasks**]  
> Wei Lu, Si-Bao Chen*, Qing-Ling Shu, Jin Tang, and Bin Luo, Senior Member, IEEE 
> 
>  *IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2024*
> 
## Introduction

The master branch is built on MMRotate which works with **PyTorch 1.6+**.

DecoupleNet backbone code is placed under mmrotate/models/backbones/, and the train/test configure files are placed under configs/decouplenet/ 


## Results and models

Imagenet 300-epoch pre-trained DecoupleNet-D0 backbone: [Download](https://github.com/lwCVer/DecoupleNet/releases/download/weights/DecoupleNet_D0.pth)

Imagenet 300-epoch pre-trained DecoupleNet_D1 backbone: [Download](https://github.com/lwCVer/DecoupleNet/releases/download/weights/DecoupleNet_D1.pth)

Imagenet 300-epoch pre-trained DecoupleNet_D2 backbone: [Download](https://github.com/lwCVer/DecoupleNet/releases/download/weights/DecoupleNet_D2.pth)

DOTA1.0

|             Model              |  mAP  | training mode | Batch Size |                                                       Configs                                                       |                                                              Download                                                               |
|:------------------------------:|:-----:|---------------|:----------:|:-------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|
| DecoupleNet_D0 (1024,1024,200) | 77.38 | single-scale  |    1\*8    | [ORCNN_DecoupleNet_D0_fpn_le90_dota10_ss_e36](./configs/DecoupleNet/ORCNN_DecoupleNet_D0_fpn_le90_dota10_ss_e36.py) |          [model](https://github.com/lwCVer/DecoupleNet/releases/download/weights/decouplenet_d0_orcnn_e36.pth)           |
| DecoupleNet_D2 (1024,1024,200) | 78.04 | single-scale  |    2\*4    | [ORCNN_DecoupleNet_D2_fpn_le90_dota10_ss_e36](./configs/DecoupleNet/ORCNN_DecoupleNet_D2_fpn_le90_dota10_ss_e36.py) |          [model](https://github.com/lwCVer/DecoupleNet/releases/download/weights/decouplenet_d2_orcnn_e36.pth)           |


DIOR-R 

|                    Model                     |  mAP  | Batch Size |
| :------------------------------------------: |:-----:| :--------: |
|                   LWGANet_L2                   | 67.08 |    1\*8    |

## Installation

```shell
conda create -n dyfrdet_soda python=3.8
conda activate dyfrdet_soda
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim==0.3.9
mim install mmcv-full==1.7.2
mim install mmdet==2.28.2
cd mmrotate-dyfrdet
pip install -r requirements/build.txt
pip install -e .
pip install timm==1.0.15
pip install tensorboard==2.14.0

"/xxx/envs/dyfrdet_soda/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py"
add 'data_batch['epoch'] = self.epoch' before 'outputs = self.model.train_step(data_batch, self.optimizer,**kwargs)'
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)


## Citation
```

```
