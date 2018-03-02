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
import json
import time
from datetime import datetime
import argparse

import numpy as np
import torch

from torch import nn
from torch import optim
from torch.autograd import Variable

from models.conv3d_repetition import Conv3D_Repetition
from transform.spatial import *
from dataset import BlenderSyntheticDataset
from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, net, data_loader, optimizer):

    examples_per_second = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for step, (clips, labels) in enumerate(data_loader):

        clips  = Variable(clips)
        labels = Variable(labels)
        if args.ngpu > 0:
            clips  = clips.cuda()
            labels = labels.cuda()

        # Forward pass through the network
        logits = net(clips)

        loss = criterion(logits, labels)
        acc = calculate_accuracy(logits, labels)

        losses.update(loss.data[0], clips.size(0))
        accuracies.update(acc, clips.size(0))

        # Perform optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Only for time measurement of step through network
        batch_examples_per_second = args.batch_size / float(time.time() - end_time)
        examples_per_second.update(batch_examples_per_second)

        print("[{}] Epoch {}. Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Accuracy = {:.2f}, Loss = {:.3f}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M"), epoch, step, len(data_loader),
            args.batch_size, examples_per_second.avg, accuracies.avg, losses.avg
        ))

        end_time = time.time()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train 3D ConvNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument('--data_path', type=str, help='Root path for dataset.')

    # # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The Learning Rate.')
    # parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    # parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    #
    # # Checkpoints
    # parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
    # parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    # parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    #
    # # Architecture
    # parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    # parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    # parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    #
    # # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--workers', type=int, default=4, help='Pre-fetching threads.')
    #
    # # i/o
    # parser.add_argument('--log', type=str, default='./', help='Log folder.')
    args = parser.parse_args()

    ############################################################################
    # Nice example: https://github.com/prlz77/ResNeXt.pytorch/blob/master/train.py

    # Define the input pipeline
    spatial_transform = Compose([ToTensor(225), Normalize([0, 0, 0], [1, 1, 1])])
    train_data = BlenderSyntheticDataset(dataset_path=args.data_path, spatial_transform=spatial_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # Define the network
    net = Conv3D_Repetition(num_classes=train_data.num_classes())
    if args.ngpu > 0: net.cuda()

    # Loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_epoch(epoch, net, train_loader, optimizer)


