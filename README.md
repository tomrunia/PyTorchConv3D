# 3D ConvNets in PyTorch

This repository contains PyTorch models of I3D and 3D-ResNets based on the following repositories:

- https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py
- https://github.com/kenshohara/3D-ResNets-PyTorch/

**Models:** I3D, 3D-ResNet, 3D-DenseNet, 3D-ResNeXt  
**Datasets:** UCF-101, Kinetics, ActivityNet, Charades

## Installation

Clone and install:

```sh
git clone https://github.com/tomrunia/PyTorchConv3D.git
cd PyTorchConv3D
pip install -r requirements.txt
python setup.py install
```

## Requirements

- Python 2.7 or 3.5+ (developed with 3.7)
- Numpy (developed with 1.15.0)
- PyTorch >= 0.4.0
- TensorboardX (optional)
- PIL (optional)

## Examples

Training ResNet-34 from scratch on UCF-101

```
python train.py --dataset=ucf101 --model=resnet --video_path=/home/tomrunia/data/UCF-101/jpg --annotation_path=/home/tomrunia/data/UCF-101/ucfTrainTestlist/ucf101_01.json --batch_size=64 --num_classes=101 --momentum=0.9 --weight_decay=1e-3 --model_depth=34 --resnet_shortcut=A --spatial_size=112 --sample_duration=16 --optimizer=SGD --learning_rate=0.01
```

## References

- Carreira and Zisserman - "[Quo Vadis,
Action Recognition?](https://arxiv.org/abs/1705.07750)" (CVPR, 2017)
- Hara _et al._ - "[Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/abs/1711.09577)" (CVPR, 2018)