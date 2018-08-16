
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

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

from epoch_iterators import train_epoch, validation_epoch
from utils.utils import *
import factory.data_factory as data_factory
import factory.model_factory as model_factory
from config import parse_opts


####################################################################
####################################################################
# Configuration and logging

config = parse_opts()
config = prepare_output_dirs(config)

print_config(config)
write_config(config, os.path.join(config.save_dir, 'config.json'))

# TensorboardX summary writer
if not config.no_tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=config.log_dir)
else:
    writer = None

####################################################################
####################################################################
# Initialize model

device = torch.device(config.device)

# Returns the network instance (I3D, 3D-ResNet etc.)
model = model_factory.get_model(config)

# Move the model to GPU memory
model = model.to(device)

####################################################################
####################################################################
# Setup of data transformations

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
# Setup of data pipeline

# Obtain 'train' and 'validation' loaders
print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))
data_loaders, datasets = data_factory.get_data_loaders(
    config, spatial_transform, temporal_transform, target_transform)
phases = ['train', 'validation'] if 'validation' in data_loaders else ['train']

####################################################################
####################################################################
# Optimizer and loss initialization

criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(config, model.parameters())

# Restore optimizer params and set config.start_index
restore_optimizer_state(config, optimizer)

# Learning rate scheduler
milestones = [int(x) for x in config.lr_scheduler_milestones.split(',')]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, config.lr_scheduler_gamma)

####################################################################
####################################################################
# Resume training from previous checkpoint

if config.resume_path:
    model_factory.model_restore_checkpoint(config, model, optimizer)

####################################################################
####################################################################

# Keep track of best validation accuracy
best_val_acc = 0.0

for epoch in range(config.start_epoch, config.num_epochs+1):

    # First 'training' phase, then 'validation' phase
    for phase in phases:

        if phase == 'train':

            # Perform one training epoch
            train_loss, train_acc, train_duration = train_epoch(
                config=config,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                data_loader=data_loaders['train'],
                epoch=epoch,
                summary_writer=writer
            )

        elif phase == 'validation':

            # Perform one training epoch
            val_loss, val_acc, val_duration = validation_epoch(
                config=config,
                model=model,
                criterion=criterion,
                device=device,
                data_loader=data_loaders['validation'],
                epoch=epoch,
                summary_writer=writer
            )

    print('#'*60)
    print('EPOCH {} SUMMARY'.format(epoch+1))
    print('Training Phase.')
    print('  Total Duration:              {}'.format(duration_to_string(train_duration)))
    print('  Average Train Loss:          {:.3f}'.format(train_loss))
    print('  Average Train Accuracy:      {:.3f}'.format(train_acc))

    if 'validation' in phases:
        print('Validation Phase.')
        print('  Total Duration:              {}'.format(duration_to_string(val_duration)))
        print('  Average Validation Loss:     {:.3f}'.format(val_loss))
        print('  Average Validation Accuracy: {:.3f}'.format(val_acc))

    if 'validation' in phases and val_acc > best_val_acc:
        checkpoint_path = os.path.join(config.checkpoint_dir, 'save_best.pth')
        save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
        print('Found new best validation accuracy: {:.3f}'.format(val_acc))
        print('Model checkpoint (best) written to:     {}'.format(checkpoint_path))
        best_val_acc = val_acc

    # Model saving
    if epoch % config.checkpoint_frequency == 0:
        checkpoint_path = os.path.join(config.checkpoint_dir, 'save_{:03d}.pth'.format(epoch+1))
        save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
        print('Model checkpoint (periodic) written to: {}'.format(checkpoint_path))
        cleanup_checkpoint_dir(config)  # remove old checkpoint files


# Dump all TensorBoard logs to disk for external processing
writer.export_scalars_to_json(os.path.join(config.save_dir, 'all_scalars.json'))
writer.close()

print('Finished training.')
