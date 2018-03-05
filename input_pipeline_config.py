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
import cv2

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision.transforms

from models.conv3d_repetition import Conv3D_Repetition
from transform.spatial import *
from dataset import BlenderSyntheticDataset
from utils import AverageMeter, calculate_accuracy

from tensorboardX import SummaryWriter

################################################################################


if __name__ == "__main__":

    data_path = "/home/tomrunia/data/VideoCountingDataset/BlenderSyntheticRandom/videos_as_dataset/"

    # Define the input pipeline
    spatial_transform = Compose([
        torchvision.transforms.ToPILImage(),
        RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=5, resample=2),
        torchvision.transforms.ColorJitter(0.3, 0.3, 0.3),
        ToTensor(225),
        Normalize([0, 0, 0], [1, 1, 1])
    ])
    train_data = BlenderSyntheticDataset(dataset_path=data_path, max_num_examples=1000, spatial_transform=spatial_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=10, shuffle=False,
        num_workers=1, pin_memory=True)

    for step, (videos, labels) in enumerate(train_loader):
        for example_idx in range(len(videos)):
            print("LabelIdx = {}, Label = {}".format(labels[example_idx], train_data.labels[labels[example_idx]]))
            video = videos[example_idx,0,].numpy()
            for frame_idx in range(len(video)):

                frame_bgr = cv2.cvtColor(video[frame_idx], code=cv2.COLOR_GRAY2BGR)
                if frame_idx % train_data.labels[labels[example_idx]] == 0:
                    cv2.circle(frame_bgr, (10, 10), 5, color=(0,255,255), thickness=-1)

                if frame_idx == 0:
                    cv2.imshow("First Frame", frame_bgr)

                cv2.imshow("Current Frame", frame_bgr)
                key = cv2.waitKey(0)
                if key == ord('n'):
                    break