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

import os
import glob

import numpy as np
import h5py

import torch
from torch.utils.data import Dataset


class BlenderSyntheticDataset(Dataset):

    def __init__(self, root_path, subset, spatial_transform=None,
                 temporal_transform=None, target_transform=None):

        assert temporal_transform is None, 'Temporal transform not supported for BlenderDataset'
        assert target_transform is None,   'Target transform not supported for BlenderDataset'
        assert subset in ('train', 'validation')

        self._root_path = root_path
        self._subset = subset

        self._spatial_transform = spatial_transform
        self._temporal_transform = temporal_transform
        self._target_transform = target_transform

        self._make_dataset()


    def _make_dataset(self):

        self._data_files = glob.glob(os.path.join(self._root_path, self._subset, 'data', '*.h5'))
        self._data_files.sort()

        with h5py.File(self._data_files[0]) as hf:
            self._num_examples_per_file = len(hf['labels'])
            self._num_examples = len(self._data_files)*self._num_examples_per_file

        self._classes = set()
        for i, path in enumerate(self._data_files):
            with h5py.File(path) as hf:
                self._classes = self._classes.union(set(list(hf['labels'].value)))

        self._target_offset = min(self._classes)

        print('  Number of {} HDF5 files found: {}'.format(self._subset, len(self._data_files)))
        print('  Number of {} examples found:   {}'.format(self._subset, len(self)))
        print('  Number of {} targets found:    {}'.format(self._subset, len(self.classes)))

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):

        container_idx = idx // self._num_examples_per_file
        example_idx = (idx-(container_idx*self._num_examples_per_file))

        with h5py.File(self._data_files[container_idx], 'r') as hf:
            clip   = hf['videos'][example_idx]
            target = hf['labels'][example_idx]

        # Apply spatial transformations (i.e. spatial cropping, flipping, normalization)
        if self._spatial_transform is not None:
            self._spatial_transform.randomize_parameters()
            clip = [self._spatial_transform(frame) for frame in clip]

        clip   = torch.stack(clip, dim=1).type(torch.FloatTensor)
        target = torch.from_numpy(np.asarray(target-self.target_offset, np.int64))
        return clip, target

    @property
    def classes(self):
        return self._classes

    @property
    def subset(self):
        return self._subset

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def target_offset(self):
        return self._target_offset