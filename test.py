# MIT License
# 
# Copyright (c) 2018 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-03-01

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import json
import time
from datetime import datetime
import argparse

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.utils
import torchvision.transforms

from models.conv3d import Conv3D_Repetition
from video_transform.spatial import *
from dataset import BlenderSyntheticDataset
from utils import AverageMeter, calculate_accuracy

from tensorboardX import SummaryWriter

################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train 3D ConvNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument('--data_path', type=str, help='Root path for dataset.')
    parser.add_argument('--checkpoint_path', type=str, default='./output/', help='Root path for dataset.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size.')

    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--workers', type=int, default=4, help='Pre-fetching threads.')

    args = parser.parse_args()

    ############################################################################

    # Define the input pipeline
    spatial_transform = Compose([
        ToTensor(225),
        Normalize([0, 0, 0], [1, 1, 1])
    ])

    dataset = BlenderSyntheticDataset(dataset_path=args.data_path, spatial_transform=spatial_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Define the network
    net = Conv3D_Repetition(num_classes=dataset.num_classes())
    if args.ngpu > 0: net.cuda()

    # This has any effect only on modules such as Dropout or BatchNorm.
    net.eval()

    # for epoch in range(args.epochs):
    #
    #     # Perform optimization for one epoch
    #     train_epoch(epoch=epoch, net=net, data_loader=train_loader, optimizer=optimizer,
    #                 summary_writer=summary_writer,
    #                 scalar_summary_interval=args.scalar_summary_interval,
    #                 image_summary_interval=args.image_summary_interval,
    #                 checkpoint_path=checkpoint_path, checkpoint_interval=args.checkpoint_interval)
    #
    #


