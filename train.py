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

from models.conv3d_repetition import Conv3D_Repetition
from dataset import BlenderSyntheticDataset
from transform.spatial import *
from utils import *

from tensorboardX import SummaryWriter

################################################################################



examples_per_second = AverageMeter(history=10)
losses = AverageMeter(history=10)
accuracies = AverageMeter(history=10)


def train(epoch, net, data_loader, optimizer, summary_writer=None,
          scalar_summary_interval=1, image_summary_interval=100):

    # This has any effect only on modules such as Dropout or BatchNorm.
    net.train()

    batches_per_epoch = len(data_loader)
    end_time = time.time()

    for step_in_batch, (clips, labels) in enumerate(data_loader):

        # Compute the global step
        step = (epoch*batches_per_epoch) + step_in_batch

        # Check the use of volatile=True and async=True
        # See here: https://github.com/pytorch/examples/blob/master/imagenet/main.py
        clips  = Variable(clips)
        labels = Variable(labels)
        if args.ngpu > 0:
            clips  = clips.cuda()
            labels = labels.cuda()

        # Forward pass through the network
        logits = net(clips)

        loss = criterion(logits, labels)
        acc = calculate_accuracy(logits, labels)

        losses.push(loss.data[0])
        accuracies.push(acc)

        # Perform optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Only for time measurement of step through network
        batch_examples_per_second = args.batch_size / float(time.time() - end_time)
        examples_per_second.push(batch_examples_per_second)

        print("[{}] Epoch {}. Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Accuracy = {:.2f}, Loss = {:.3f}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M"), epoch+1, step_in_batch, len(data_loader),
            args.batch_size, examples_per_second.average(), accuracies.average(), losses.average()
        ))

        # Save to TensorBoard
        if step % scalar_summary_interval == 0:
            summary_writer.add_scalar('train/loss', loss.data[0], step)
            summary_writer.add_scalar('train/accuracy', acc, step)
            summary_writer.add_scaler('train/examples_per_second', batch_examples_per_second, step)

        if step % image_summary_interval == 0:
            image_sequence = clips[0].permute(1,0,2,3)
            image_grid = torchvision.utils.make_grid(image_sequence.data, nrow=8)
            summary_writer.add_image('train/images', image_grid)

        end_time = time.time()


################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train 3D ConvNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument('--data_path', type=str, help='Root path for dataset.')
    parser.add_argument('--output_path', type=str, default='./output/', help='Root path for dataset.')

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The Learning Rate.')

    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs. Set to 0 to perform on CPU.')
    parser.add_argument('--workers', type=int, default=6, help='Pre-fetching threads.')

    # Logging
    parser.add_argument('--scalar_summary_interval', type=int, default=10, help='Scalar summary saving frequency (steps).')
    parser.add_argument('--image_summary_interval', type=int, default=10, help='Image summary saving frequency (steps).')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Checkpoint saving frequency (epochs).')

    args = parser.parse_args()

    ############################################################################
    # Nice example: https://github.com/prlz77/ResNeXt.pytorch/blob/master/train.py
    # Another one:  https://github.com/pytorch/examples/blob/master/imagenet/main.py

    run_output_path = os.path.join(args.output_path, datetime.now().strftime("%Y%m%d_%H:%M"))
    os.makedirs(run_output_path)

    checkpoint_path = os.path.join(run_output_path, 'checkpoints')
    summary_path    = os.path.join(run_output_path, 'summaries')

    # Initialize TensorBoard summary writer
    summary_writer = SummaryWriter(summary_path)

    ############################################################################

    # Define the input pipeline
    spatial_transform = Compose([
        torchvision.transforms.ToPILImage(),
        RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=5, resample=2),
        torchvision.transforms.ColorJitter(0.3, 0.3, 0.3),
        ToTensor(225),
        Normalize([0, 0, 0], [1, 1, 1])
    ])

    # TODO: checkout best way train/validation data loading
    train_data = BlenderSyntheticDataset(dataset_path=args.data_path, spatial_transform=spatial_transform)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    ############################################################################

    # Define the network
    # TODO: check how to use weight decay ...
    net = Conv3D_Repetition(num_classes=train_data.num_classes())
    if args.ngpu > 0: net.cuda()

    # Loss criterion and optimizer
    # TODO: check how to setup learning rate decay...
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate)

    ############################################################################
    # Main train/evaluation loop

    best_val_loss = np.inf
    best_val_acc  = 0.0

    for epoch in range(args.epochs):

        # Perform optimization for one epoch
        train(epoch=epoch, net=net, data_loader=train_loader,
              optimizer=optimizer, summary_writer=summary_writer,
              scalar_summary_interval=args.scalar_summary_interval,
              image_summary_interval=args.image_summary_interval)

        # Perform validation for entire dataset
        # validation_error = ...
        epoch_val_loss = np.random.rand()
        epoch_val_acc  = np.random.rand()
        is_best = epoch_val_acc > best_val_acc

        # Save the model after each epoch
        if checkpoint_path is not None and epoch % args.checkpoint_interval == 0:
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            save_file_path = os.path.join(checkpoint_path, "save_{:06d}.pth.tar".format(epoch+1))
            states = {
                'epoch':      epoch+1,
                'state_dict': net.state_dict(),
                'optimizer':  optimizer.state_dict()
            }
            save_checkpoint(states, is_best, save_file_path)
            print("[{}] Saved model checkpoint: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M"), save_file_path))

        # Keep track of the best validation performance
        if epoch_val_acc > best_val_acc:
            best_val_acc  = epoch_val_acc
            best_val_loss = epoch_val_loss

    # Save JSON file of scalars to disk
    summary_writer.export_scalars_to_json(os.path.join(args.out_path, 'train_summary.json'))
    summary_writer.close()


