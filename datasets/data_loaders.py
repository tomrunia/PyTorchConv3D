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
# Date Created: 2018-08-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torchvision

##################################################
#   This is unfinished and currently not used.   #
##################################################

def init_datasets(data_path, valid_frac, batch_size, num_workers,
                  shuffle_initial=False, shuffle_seed=1234):

    print("#"*80)
    print("Initializing data loaders... ")

    # Define the input pipeline
    train_transform = Compose([
        torchvision.transforms.ToPILImage(),
        RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=5, resample=2),
        torchvision.transforms.ColorJitter(0.3, 0.3, 0.3),
        ToTensor(225),
        Normalize([0, 0, 0], [1, 1, 1])
    ])

    valid_transform = Compose([
        torchvision.transforms.ToPILImage(),
        ToTensor(225),
        Normalize([0, 0, 0], [1, 1, 1])
    ])

    # Initialize training and validation dataset
    train_data = BlenderSyntheticDataset(data_path, spatial_transform=train_transform)
    valid_data = BlenderSyntheticDataset(data_path, spatial_transform=valid_transform)

    # Determine which examples to use for training and validation
    num_examples = len(train_data)
    indices = np.arange(0, num_examples)
    split = int(np.floor(valid_frac * num_examples))

    if shuffle_initial:
        np.random.seed(shuffle_seed)
        np.random.shuffle(indices)

    # Set the indices for training and validation
    train_idx = indices[split:]
    valid_idx = indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # And finally initialize the dataset loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=True)

    print("Validation Fraction:     {:.2f}".format(valid_frac))
    print("Num Train examples:      {}".format(len(train_idx)))
    print("Num Validation examples: {}".format(len(valid_idx)))
    print("#"*80)

    return train_loader, valid_loader, train_data.labels