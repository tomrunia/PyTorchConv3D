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
import glob
import shutil
import time

import simplejson

import numpy as np

import torch
import torch.optim
import torchvision

####################################################################
####################################################################

def print_config(config):
    print('#'*60)
    print('Training configuration:')
    for k,v  in vars(config).items():
        print('  {:>20} {}'.format(k, v))
    print('#'*60)

def write_config(config, json_path):
    with open(json_path, 'w') as f:
        f.write(simplejson.dumps(vars(config), indent=4, sort_keys=True))

def output_subdir(config):
    prefix = time.strftime('%Y%m%d_%H%M')
    subdir = "{}_{}_{}_lr{:.3f}".format(prefix, config.dataset, config.model, config.learning_rate)
    return os.path.join(config.save_dir, subdir)

def init_cropping_scales(config):
    # Determine cropping scales
    config.scales = [config.initial_scale]
    for i in range(1, config.num_scales):
        config.scales.append(config.scales[-1] * config.scale_step)
    return config

def set_lr_scheduling_policy(config):
    if config.lr_plateau_patience > 0 and not config.no_eval:
        config.lr_scheduler = 'plateau'
    else:
        config.lr_scheduler = 'multi_step'
    return config

def prepare_output_dirs(config):
    # Set output directories
    config.save_dir = output_subdir(config)
    config.checkpoint_dir = os.path.join(config.save_dir, 'checkpoints')
    config.log_dir = os.path.join(config.save_dir, 'logs')

    # And create them
    if os.path.exists(config.save_dir):
        # Only occurs when experiment started the same minute
        shutil.rmtree(config.save_dir)

    os.mkdir(config.save_dir)
    os.mkdir(config.checkpoint_dir)
    os.mkdir(config.log_dir)
    return config

def cleanup_checkpoint_dir(config):
    checkpoint_files = glob.glob(os.path.join(config.checkpoint_dir, 'save_*.pth'))
    checkpoint_files.sort()
    if len(checkpoint_files) > config.checkpoints_num_keep:
        os.remove(checkpoint_files[0])

def duration_to_string(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))

####################################################################
####################################################################

def get_optimizer(config, params):
    if config.optimizer == 'SGD':
        return torch.optim.SGD(params, config.learning_rate, config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        return torch.optim.RMSprop(params, config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        return torch.optim.Adam(params, config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError('Chosen optimizer is not supported, please choose from (SGD | adam | rmsprop)')

def restore_optimizer_state(config, optimizer):
    if not config.checkpoint_path: return
    checkpoint = torch.load(config.checkpoint_path)
    if 'optimizer' in checkpoint.keys():
        # I3D model has no optimizer state
        config.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('WARNING: not restoring optimizer state as it is not found in the checkpoint file.')

def current_learning_rate(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

def current_weight_decay(optimizer):
    return optimizer.state_dict()['param_groups'][0]['weight_decay']

def save_checkpoint(save_file_path, epoch, model_state_dict, optimizer_state_dict):
    states = {'epoch': epoch+1, 'state_dict': model_state_dict, 'optimizer':  optimizer_state_dict}
    torch.save(states, save_file_path)

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value

def write_clips_as_grid(step, clips, targets, dataset, output_dir='./output/', num_examples=10, n_row=4):
    for i in range(0,num_examples):
        clip = clips[i].permute((1,0,2,3))
        grid = torchvision.utils.make_grid(clip, nrow=n_row, normalize=True)
        class_label = dataset.class_names[int(targets[i].numpy())]
        torchvision.utils.save_image(grid, os.path.join(output_dir, 'step{:04d}_example{:02d}_{}.jpg'.format(step+1, i, class_label)))