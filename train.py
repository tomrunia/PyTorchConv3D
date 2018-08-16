
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
import time
from datetime import datetime

import torch.nn as nn

from transforms.spatial_transforms import Compose, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

from utils.utils import *
from factory.data_factory import *
from factory.model_factory import generate_model
from config import parse_opts

from tensorboardX import SummaryWriter

####################################################################
####################################################################

config = parse_opts()
config = prepare_output_dirs(config)

print_config(config)
write_config(config, os.path.join(config.save_dir, 'config.json'))

# TensorboardX summary writer
writer = SummaryWriter(log_dir=config.log_dir)

####################################################################
####################################################################

device = torch.device(config.device)

print('[{}] Initializing {} model (num_classes={})...'.format(datetime.now().strftime("%A %H:%M"), config.model, config.num_classes))

# Returns the network instance (I3D, 3D-ResNet etc.)
model, params_to_train = generate_model(config)  # TODO: check what goes wrong here
model = model.to(device)

####################################################################
####################################################################

# Determine cropping scales
config.scales = [config.initial_scale]
for i in range(1, config.num_scales):
    config.scales.append(config.scales[-1] * config.scale_step)

spatial_transform = Compose([
    MultiScaleRandomCrop(config.scales, config.spatial_size),
    RandomHorizontalFlip(),
    ToTensor(config.norm_value),
    Normalize([0, 0, 0], [1, 1, 1])
])

temporal_transform = TemporalRandomCrop(config.sample_duration)
target_transform   = ClassLabel()

####################################################################
####################################################################

# Obtain 'train' and 'validation' loaders
print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))
data_loaders, datasets = get_data_loaders(config, spatial_transform, temporal_transform, target_transform)
phases = ['train', 'validation'] if 'validation' in data_loaders else ['train']

####################################################################
####################################################################

examples_per_second = AverageMeter(config.history_steps)
losses = AverageMeter(config.history_steps)
accuracies = AverageMeter(config.history_steps)

####################################################################
####################################################################

# TODO: why does params_to_train instead of model.parameters() not work?
criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(config, model.parameters())

# Learning rate scheduler
milestones = [int(x) for x in config.lr_scheduler_milestones.split(',')]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, config.lr_scheduler_gamma)

####################################################################
####################################################################
# Good example: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

train_steps_per_epoch = int(np.ceil(len(datasets['train'])/config.batch_size))

for epoch in range(config.num_epochs):

    # First 'training' phase, then 'validation' phase
    for phase in phases:

        print('#'*60)
        print('Starting {} epoch {}'.format(epoch+1, phase))

        if phase == 'train':
            model.train()
        else:
            model.eval()
            val_losses = []
            val_accuracies = []

        for step, (clips, targets) in enumerate(data_loaders[phase]):

            start_time = time.time()

            # Prepare for next iteration
            optimizer.zero_grad()

            # Move inputs to GPU memory
            clips   = clips.to(device)
            targets = targets.to(device)

            # Feed-forward through the network
            logits = model.forward(clips)

            # Only for ResNet etc. [n_batch,400] => [n_batch,400,1]
            #if logits.dim() == 2:
            #   logits = logits.unsqueeze(-1)

            _, preds = torch.max(logits, 1)
            loss = criterion(logits, targets)

            # Calculate accuracy
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / config.batch_size

            # Calculate elapsed time for this step
            step_time = config.batch_size/float(time.time() - start_time)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            if phase == 'train':

                global_step = (epoch*train_steps_per_epoch) + step

                # Update average statistics
                examples_per_second.push(step_time)
                losses.push(loss.item())
                accuracies.push(accuracy.item())

                if step % config.print_frequency == 0:
                    print("[{}] Epoch {}. Batch {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                              "LR = {:.3f}, Accuracy = {:.3f}, Loss = {:.3f}".format(
                            datetime.now().strftime("%A %H:%M"), epoch+1, step,
                            train_steps_per_epoch, examples_per_second.average(),
                            current_learning_rate(optimizer), accuracies.average(), loss))

                if step % config.log_frequency == 0:
                    writer.add_scalar('train/loss', loss, global_step)
                    writer.add_scalar('train/accuracy', accuracy, global_step)
                    writer.add_scalar('train/examples_per_second', step_time, global_step)
                    writer.add_scalar('train/learning_rate', current_learning_rate(optimizer), global_step)
                    writer.add_scalar('train/weight_decay', current_weight_decay(optimizer), global_step)

            else:

                print("[{}] Epoch {}. Validation Batch {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                          "Accuracy = {:.3f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%A %H:%M"), epoch+1, step,
                        len(datasets[phase])//config.batch_size,
                        step_time, accuracy.item(), loss.item()))

        if phase == 'train':
            scheduler.step(epoch)

        if epoch % config.checkpoint_frequency == 0:
            save_checkpoint_path = os.path.join(config.checkpoint_dir, 'save_{:03}.pth'.format(epoch))
            print('Checkpoint written to: {}'.format(save_checkpoint_path))
            save_checkpoint(save_checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())

writer.export_scalars_to_json(os.path.join(config.save_dir, 'all_scalars.json'))
writer.close()

print('Finished training.')
