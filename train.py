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


import time
from datetime import datetime
import argparse

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torchvision.utils

from models.conv3d import Conv3D_Repetition
from dataset import init_datasets
from utils import *

from tensorboardX import SummaryWriter

################################################################################

history = 25

examples_per_second = AverageMeter(history=history)
losses = AverageMeter(history=history)
accuracies = AverageMeter(history=history)


def train(epoch, net, criterion, data_loader, optimizer, learning_rate_scheduler=None,
          summary_writer=None, scalar_summary_interval=1, image_summary_interval=100):

    net.train()

    # Update learning rate decay
    learning_rate_scheduler.step(epoch)

    batches_per_epoch = len(data_loader)
    end_time = time.time()

    for step_in_batch, (inputs, labels) in enumerate(data_loader):

        # Compute the global step
        step = (epoch*batches_per_epoch) + step_in_batch

        # Check the use of volatile=True and async=True
        # See here: https://github.com/pytorch/examples/blob/master/imagenet/main.py
        inputs = Variable(inputs)
        labels = Variable(labels)
        if args.num_gpu > 0:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Forward pass through the network
        logits = net(inputs)

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

        curr_learning_rate = learning_rate_scheduler.get_lr()[0]

        print("[{}] Epoch {}, Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
              "LR = {:.4f}, Accuracy = {:.3f}, Loss = {:.3f}".format(
            datetime.now().strftime("%A %H:%M"), epoch+1, step_in_batch, len(data_loader),
            args.batch_size, examples_per_second.average(), curr_learning_rate,
            accuracies.average(), losses.average()
        ))

        # Save to TensorBoard
        if step % scalar_summary_interval == 0:
            summary_writer.add_scalar('train/loss', loss.data[0], step)
            summary_writer.add_scalar('train/accuracy', acc, step)
            summary_writer.add_scalar('train/examples_per_second', batch_examples_per_second, step)
            summary_writer.add_scalar('train/learning_rate', curr_learning_rate, epoch_first_step)

        if step % image_summary_interval == 0:
            image_sequence = inputs[0].permute(1,0,2,3)
            image_grid = torchvision.utils.make_grid(image_sequence.data, nrow=8)
            summary_writer.add_image('train/images', image_grid)

        end_time = time.time()

def validate(net, criterion, data_loader):

    net.eval()

    num_batches = len(data_loader)
    epoch_losses = []
    epoch_accuracies = []

    end_time = time.time()
    print("#"*60)

    for valid_step, (inputs, labels) in enumerate(data_loader):

        inputs = Variable(inputs)
        labels = Variable(labels)
        if args.num_gpu > 0:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Forward pass through the network
        logits = net(inputs)

        # Calculate and save metrics
        loss = criterion(logits, labels)
        acc = calculate_accuracy(logits, labels)
        epoch_losses.append(loss.data[0])
        epoch_accuracies.append(acc)

        # Only for time measurement of step through network
        batch_examples_per_second = args.batch_size / float(time.time() - end_time)
        examples_per_second.push(batch_examples_per_second)

        print("[{}] Performing validation {:04d}/{:04d}, Examples/Sec = {:.2f}, "
              "Accuracy = {:.3f}, Loss = {:.3f}".format(
            datetime.now().strftime("%A %H:%M"), valid_step, num_batches,
            examples_per_second.average(), acc, loss.data[0]
        ))

        # Save one validation example from the first batch
        if valid_step == 0:
            example_idx = np.random.randint(len(inputs))
            image_sequence = inputs[example_idx].permute(1,0,2,3)
            image_grid = torchvision.utils.make_grid(image_sequence.data, nrow=8)
            summary_writer.add_image('valid/images', image_grid)

        end_time = time.time()

    val_loss = float(np.mean(epoch_losses))
    val_acc  = float(np.mean(epoch_accuracies))

    print("VALIDATION SUMMARY ({} batches):".format(len(data_loader)))
    print("  Loss:     {:.3f}".format(val_loss))
    print("  Accuracy: {:.3f}".format(val_acc))
    print("#"*60)

    return val_loss, val_acc

################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train 3D ConvNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument('--data_path', type=str, required=True, help='Root path for dataset.')
    parser.add_argument('--output_path', type=str, default='./output/', help='Root path for dataset.')

    # Optimization options
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--valid_frac', type=float, default=0.1, help='Fraction of dataset to use for validation.')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.1, help='Learning rate decay factor.')
    parser.add_argument('--learning_rate_decay_epochs', type=int, default=20, help='After how many epochs to decay learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay on trainable parameters.')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='Dropout rate for last fully-connected layer.')

    # Acceleration
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs. Set to 0 to perform on CPU.')
    parser.add_argument('--workers', type=int, default=8, help='Pre-fetching threads.')

    # Logging
    parser.add_argument('--scalar_summary_interval', type=int, default=10, help='Scalar summary saving frequency (steps).')
    parser.add_argument('--image_summary_interval', type=int, default=10, help='Image summary saving frequency (steps).')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Checkpoint saving frequency (epochs).')

    args = parser.parse_args()

    ############################################################################

    run_output_path = os.path.join(args.output_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_output_path)

    checkpoint_path = os.path.join(run_output_path, 'checkpoints')
    summary_path    = os.path.join(run_output_path, 'summaries')

    # Initialize TensorBoard summary writer
    summary_writer = SummaryWriter(summary_path)

    ############################################################################

    train_loader, valid_loader, classes = init_datasets(
        data_path=args.data_path, batch_size=args.batch_size,
        valid_frac=args.valid_frac, num_workers=args.workers,
        shuffle_initial=False)

    ############################################################################

    # Define the network
    net = Conv3D_Repetition(num_classes=len(classes))
    if args.num_gpu > 0: net.cuda()

    # Loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.RMSprop(
        params=net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Setup learning rate decay
    learning_rate_scheduler = StepLR(optimizer=optimizer,
                                     step_size=args.learning_rate_decay_epochs,
                                     gamma=args.learning_rate_decay_factor)

    ############################################################################
    # Main train/evaluation loop

    best_val_loss = np.inf
    best_val_acc  = 0.0

    for epoch in range(args.epochs):

        epoch_first_step = epoch*len(train_loader)

        # Perform optimization for one epoch
        train(epoch=epoch, net=net, criterion=criterion,
              data_loader=train_loader, optimizer=optimizer,
              learning_rate_scheduler=learning_rate_scheduler,
              summary_writer=summary_writer,
              scalar_summary_interval=args.scalar_summary_interval,
              image_summary_interval=args.image_summary_interval)

        # Perform evaluation over entire validation set
        epoch_val_loss, epoch_val_acc = validate(net=net,
                                                 criterion=criterion,
                                                 data_loader=valid_loader)

        # Save validation performance as summaries
        summary_writer.add_scalar('validation/loss', epoch_val_loss, epoch_first_step)
        summary_writer.add_scalar('validation/accuracy', epoch_val_acc, epoch_first_step)
        is_best = epoch_val_acc > best_val_acc

        # Save the model after each epoch
        if checkpoint_path is not None and epoch % args.checkpoint_interval == 0:
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            save_file_path = os.path.join(checkpoint_path, "checkpoint_{:06d}.pth.tar".format(epoch+1))
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
    summary_writer.export_scalars_to_json(os.path.join(args.output_path, 'train_summary.json'))
    summary_writer.close()


