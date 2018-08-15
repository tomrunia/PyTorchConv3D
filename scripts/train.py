
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor

#from transforms.temporal_transforms import LoopPadding, TemporalRandomCrop
#from transforms.target_transforms import ClassLabel, VideoID
#from transforms.target_transforms import Compose as TargetCompose

from models.i3d import InceptionI3d
from datasets.blender_synthetic import BlenderSyntheticDataset
from utils.utils import AverageMeter, get_optimizer, print_config
from utils.config import parse_opts

####################################################################
####################################################################

config = parse_opts()
print_config(config)

####################################################################
####################################################################

device = torch.device(config.device)

print('[{}] Initializing I3D model (num_classes={})...'.format(datetime.now().strftime("%A %H:%M"), config.num_classes))
net = InceptionI3d(num_classes=config.num_classes, in_channels=3, dropout_keep_prob=config.dropout_keep_prob)

if config.checkpoint_path:
    print('[{}] Restoring pretrained weights from: {}...'.format(datetime.now().strftime("%A %H:%M"), config.checkpoint_path))
    model_dict = net.state_dict()
    checkpoint_state_dict = torch.load(config.checkpoint_path)
    checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_dict}
    net.load_state_dict(checkpoint_state_dict)

    # Print the layer names of restored variables
    layer_names = set([k.split('.')[0] for k in checkpoint_state_dict.keys()])
    print('  Restored weights: {}'.format(layer_names))

# Disabling finetuning for all layers
net.freeze_weights()

# Replace last layer with different number of logits when finetuning
if config.num_classes != config.num_finetune_classes:
    print('[{}] Changing logits size for finetuning (num_classes={})...'.format(
        datetime.now().strftime("%A %H:%M"), config.num_finetune_classes))
    net.replace_logits(config.num_finetune_classes)

# Enable gradient for layers to finetune
finetune_prefixes = config.finetune_prefixes.split(',')
net.set_finetune_layers(finetune_prefixes)

# Obtain parameters to be fed into the optimizer
params_to_train = net.trainable_params()
net = net.to(device)

####################################################################
####################################################################

spatial_transform = Compose([
    RandomHorizontalFlip(),
    ToTensor(config.norm_value),
    Normalize([0, 0, 0], [1, 1, 1])
])
temporal_transform = None  # TemporalRandomCrop(temporal_size)
target_transform   = None  # ClassLabel()

print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

dataset = BlenderSyntheticDataset(
    root_path=config.data_path,
    spatial_size=config.spatial_size,
    temporal_size=config.temporal_size,
    spatial_transform=spatial_transform,
    temporal_transform=temporal_transform,
    target_transform=target_transform)

train_loader = DataLoader(
    dataset=dataset, batch_size=config.batch_size, shuffle=True,
    num_workers=config.num_workers, pin_memory=True)

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
phase = 'train'  # TODO: add 'validation'

for epoch in range(config.num_epochs):

    print('#'*60)
    print('Starting epoch {}'.format(epoch+1))

    net.train()

    for step, (clips, targets) in enumerate(train_loader):

        start_time = time.time()

        # Prepare for next iteration
        optimizer.zero_grad()

        # Move inputs to GPU memory
        clips   = clips.to(device)
        targets = targets.to(device)

        # Feed-forward through the network
        logits = net.forward(clips)
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

        print("[{}] Epoch {}. Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.3f}, Loss = {:.3f}".format(
                datetime.now().strftime("%A %H:%M"), epoch+1, step, len(dataset)//config.batch_size,
                examples_per_second.average(), accuracies.average(),
                losses.average()))

    save_checkpoint_path = os.path.join(config.save_model_path, 'model_{:03}.ptk'.format(epoch))
    print('Saving checkpoint to: {}'.format(save_checkpoint_path))
    torch.save(net.state_dict(), save_checkpoint_path)

print('Finished training.')
