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
# Date Created: 2018-08-14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):

    def __init__(self, spatial_size, temporal_size, spatial_transform=None, temporal_transform=None, target_transform=None):
        self._spatial_size = spatial_size
        self._temporal_size = temporal_size

        self._spatial_transform = spatial_transform
        self._temporal_transform = temporal_transform
        self._target_transform = target_transform

        # work in progress
        assert self._temporal_transform is None

    def __len__(self):
        return int(1e8)

    def __getitem__(self, idx):

        shape = (self._temporal_size,self._spatial_size,self._spatial_size,3)
        clip = np.random.uniform(0, 255, shape).astype(np.uint8)

        # Apply temporal transformations (i.e. temporal cropping, looping)
        #if self._temporal_transform is not None:
        #    frame_indices = self._temporal_transform(frame_indices)

        # Apply spatial transformations (i.e. spatial cropping, flipping, normalization)
        if self._spatial_transform is not None:
            self._spatial_transform.randomize_parameters()
            clip = [self._spatial_transform(frame) for frame in clip]

        clip   = torch.stack(clip, dim=1).type(torch.FloatTensor)
        target = torch.LongTensor(1).random_(0, 10)

        return clip, target