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

import h5py
import numpy as np
import cortex.utils

import torch

import torch.utils.data
import torchvision.transforms
from torch.utils.data.sampler import SubsetRandomSampler
from transform.spatial import *



class BlenderSyntheticDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, max_num_examples=None,
                 temporal_transform=None, spatial_transform=None):

        super(BlenderSyntheticDataset, self).__init__()

        # List all data files and obtain number of examples
        self.dataset_path = dataset_path
        self.data_files = cortex.utils.find_files(dataset_path, "h5")
        self.labels = set()

        self.num_examples = self._scan_dataset_length()
        self.labels = np.asarray(list(self.labels), dtype=int)

        self.temporal_transform = temporal_transform
        self.spatial_transform  = spatial_transform

        if max_num_examples:
            self.num_examples = min(self.num_examples, max_num_examples)

    def _scan_dataset_length(self):
        num_examples = 0
        for i, data_file in enumerate(self.data_files):
            hf = h5py.File(data_file, 'r')
            labels = hf['labels']
            self.labels = self.labels.union(set(labels))
            num_examples += len(labels)
            if i == 0:
                self.examples_per_file = num_examples
            hf.close()
        return num_examples

    def num_classes(self):
        return len(self.labels)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Compute which container has the example
        file_idx  = int(np.floor(idx/self.examples_per_file))
        # Compute the index inside the container
        video_idx = idx-(file_idx*self.examples_per_file)
        with h5py.File(self.data_files[file_idx]) as hf:
            clip  = hf['videos'][video_idx]
            label = hf['labels'][video_idx]

        # Execute temporal transformations (shuffling, repeating etc...)
        # ...

        # Execute spatial transformations
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip_preproc = []
            for frame in clip:
                frame = np.expand_dims(frame, -1)
                clip_preproc.append(self.spatial_transform(frame))
        clip_preproc = torch.stack(clip_preproc, 0).permute(1,0,2,3)

        # Convert to class index
        label = int(np.argwhere(self.labels == label))
        return clip_preproc, label


################################################################################


def init_datasets(data_path, valid_frac, batch_size, num_workers,
                  shuffle_initial=False, shuffle_seed=1234):

    print("#"*60)
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
    train_data = BlenderSyntheticDataset(dataset_path=data_path, spatial_transform=train_transform)
    valid_data = BlenderSyntheticDataset(dataset_path=data_path, spatial_transform=valid_transform)

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
    print("#"*60)

    return train_loader, valid_loader, train_data.labels


