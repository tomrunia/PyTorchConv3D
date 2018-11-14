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
# Date Created: 2018-XX-XX

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from utils.utils import *

import torch


def train_epoch(config, model, criterion, optimizer, device,
                data_loader, epoch, summary_writer=None):

    print('#'*60)
    print('Epoch {}. Starting with training phase.'.format(epoch+1))

    model.train()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()

    for step, (clips, targets) in enumerate(data_loader):

        start_time = time.time()

        # Prepare for next iteration
        optimizer.zero_grad()

        # Move inputs to GPU memory
        clips = clips.to(device)
        targets = targets.to(device)
        if config.model == 'i3d':
            targets = torch.unsqueeze(targets, -1)

        # Feed-forward through the network
        logits = model.forward(clips)

        if epoch == 0 and step == 0:
            # Sanity check
            if config.checkpoint_path:
                if logits.shape[1] != config.finetune_num_classes:
                    raise RuntimeError('Number of output logits ({}) does not match number of finetune classes ({})'.format(logits.shape[1], config.finetune_num_classes))
            else:
                if logits.shape[1] != config.num_classes:
                    raise RuntimeError('Number of output logits ({}) does not match number of classes ({})'.format(logits.shape[1], config.finetune_num_classes))

        _, preds = torch.max(logits, 1)
        loss = criterion(logits, targets)

        # Calculate accuracy
        correct = torch.sum(preds == targets.data)
        accuracy = correct.double() / config.batch_size

        # Calculate elapsed time for this step
        examples_per_second = config.batch_size/float(time.time() - start_time)

        # Back-propagation and optimization step
        loss.backward()
        optimizer.step()

        # Save statistics
        accuracies[step] = accuracy.item()
        losses[step] = loss.item()

        # Compute the global step, only for logging
        global_step = (epoch*steps_in_epoch) + step

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}. Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "LR = {:.4f}, Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch+1,
                    step, steps_in_epoch, examples_per_second,
                    current_learning_rate(optimizer), accuracies[step], losses[step]))

        if summary_writer and step % config.log_frequency == 0:
            summary_writer.add_scalar('train/loss', losses[step], global_step)
            summary_writer.add_scalar('train/accuracy', accuracies[step], global_step)
            summary_writer.add_scalar('train/examples_per_second', examples_per_second, global_step)
            summary_writer.add_scalar('train/learning_rate', current_learning_rate(optimizer), global_step)
            summary_writer.add_scalar('train/weight_decay', current_weight_decay(optimizer), global_step)

        if summary_writer and step % config.log_image_frequency == 0:
            # TensorboardX video summary
            for example_idx in range(4):
                clip_for_display = clips[example_idx].clone().cpu()
                min_val = float(clip_for_display.min())
                max_val = float(clip_for_display.max())
                clip_for_display.clamp_(min=min_val, max=max_val)
                clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)
                summary_writer.add_video('train_clips/{:04d}'.format(example_idx), clip_for_display.unsqueeze(0), global_step)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc  = np.mean(accuracies)

    if summary_writer:
        summary_writer.add_scalar('train/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('train/epoch_avg_accuracy', epoch_avg_acc, epoch)

    return epoch_avg_loss, epoch_avg_acc, epoch_duration


####################################################################
####################################################################


def validation_epoch(config, model, criterion, device, data_loader, epoch, summary_writer=None):

    print('#'*60)
    print('Epoch {}. Starting with validation phase.'.format(epoch+1))

    model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()

    for step, (clips, targets) in enumerate(data_loader):

        start_time = time.time()

        # Move inputs to GPU memory
        clips   = clips.to(device)
        targets = targets.to(device)

        # Feed-forward through the network
        logits = model.forward(clips)

        _, preds = torch.max(logits, 1)
        loss = criterion(logits, targets)

        # Calculate accuracy
        correct = torch.sum(preds == targets.data)
        accuracy = correct.double() / config.batch_size

        # Calculate elapsed time for this step
        examples_per_second = config.batch_size/float(time.time() - start_time)

        # Save statistics
        accuracies[step] = accuracy.item()
        losses[step] = loss.item()

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch+1,
                    step, steps_in_epoch, examples_per_second,
                    accuracies[step], losses[step]))

        if summary_writer and step == 0:
            # TensorboardX video summary
            for example_idx in range(4):
                clip_for_display = clips[example_idx].clone().cpu()
                min_val = float(clip_for_display.min())
                max_val = float(clip_for_display.max())
                clip_for_display.clamp_(min=min_val, max=max_val)
                clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)
                summary_writer.add_video('validation_clips/{:04d}'.format(example_idx), clip_for_display.unsqueeze(0), epoch*steps_in_epoch)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc  = np.mean(accuracies)

    if summary_writer:
        summary_writer.add_scalar('validation/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('validation/epoch_avg_accuracy', epoch_avg_acc, epoch)

    return epoch_avg_loss, epoch_avg_acc, epoch_duration