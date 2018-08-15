
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

#from transforms.temporal_transforms import LoopPadding, TemporalRandomCrop
#from transforms.target_transforms import ClassLabel, VideoID
#from transforms.target_transforms import Compose as TargetCompose

from utils.utils import *
from factory.data_factory import *
from factory.model_factory import generate_model
from config import parse_opts

####################################################################
####################################################################

config = parse_opts()
print_config(config)

####################################################################
####################################################################

device = torch.device(config.device)

print('[{}] Initializing {} model (num_classes={})...'.format(datetime.now().strftime("%A %H:%M"), config.model, config.num_classes))

# Returns the network instance (I3D, 3D-ResNet etc.)
model, params_to_train = generate_model(config)
model = model.to(device)

####################################################################
####################################################################

spatial_transform = Compose([
    RandomHorizontalFlip(),
    ToTensor(config.norm_value),
    Normalize([0, 0, 0], [1, 1, 1])
])
temporal_transform = None  # TemporalRandomCrop(temporal_size)
target_transform   = None  # ClassLabel()

# Obtain 'train' and 'validation' loaders
print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))
data_loaders = get_data_loaders(config, spatial_transform, temporal_transform, target_transform)
phases = ['train', 'validation'] if 'validation' in data_loaders else ['train']

####################################################################
####################################################################

examples_per_second = AverageMeter(config.history_steps)
losses = AverageMeter(config.history_steps)
accuracies = AverageMeter(config.history_steps)

####################################################################
####################################################################

criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(
    params_to_train, config.optimizer, config.learning_rate,
    config.momentum, config.weight_decay)

####################################################################
####################################################################
# Good example: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

for epoch in range(config.num_epochs):

    # First 'training' phase, then 'validation' phase
    for phase in phases:

        print('#'*60)
        print('Starting {} epoch {}'.format(epoch+1, phase))

        if phase == 'train':
            model.train()
        else:
            model.eval()

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
            if logits.dim() == 2:
                logits = logits.unsqueeze(-1)

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

            # Update average statistics
            examples_per_second.push(step_time)
            losses.push(loss.item())
            accuracies.push(accuracy.item())

            print("[{}] Epoch {}. Batch {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch+1, step, len(data_loaders[phase])//config.batch_size,
                    examples_per_second.average(), accuracies.average(),
                    losses.average()))

        save_checkpoint_path = os.path.join(config.save_path, 'save_{:03}.pth'.format(epoch))
        print('Checkpoint written to: {}'.format(save_checkpoint_path))
        save_checkpoint(save_checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())


print('Finished training.')
