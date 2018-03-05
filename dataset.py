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
import torch.utils.data as data


class BlenderSyntheticDataset(data.Dataset):

    def __init__(self, dataset_path, max_num_examples=None,
                 temporal_transform=None, spatial_transform=None):

        super(BlenderSyntheticDataset, self).__init__()

        # List all data files and obtain number of examples
        self.dataset_path = dataset_path
        self.data_files   = cortex.utils.find_files(dataset_path, "h5")
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