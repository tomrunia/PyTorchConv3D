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
from torch.utils.data import Dataset, DataLoader


class BlenderSyntheticDataset(Dataset):

    def __init__(self, root_path, max_num_examples=None, examples_per_file=200):

        super(BlenderSyntheticDataset, self).__init__()

        # List all data files and obtain number of examples
        self.root_path = root_path
        self.data_files = cortex.utils.find_files(root_path, "h5")
        self.num_examples = self._scan_dataset_length()

        if max_num_examples:
            self.num_examples = min(self.num_examples, max_num_examples)

    def _scan_dataset_length(self):
        num_examples = 0
        for i, data_file in enumerate(self.data_files):
            hf = h5py.File(data_file, 'r')
            num_examples += len(hf['labels'])
            if i == 0:
                self.examples_per_file = num_examples
            hf.close()
        return num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Compute which container has the example
        file_idx  = int(np.floor(idx/self.examples_per_file))
        # Compute the index inside the container
        video_idx = idx-(file_idx*self.examples_per_file)
        with h5py.File(self.data_files[file_idx]) as hf:
            frames = hf['videos'][video_idx]
            label  = hf['labels'][video_idx]
        return frames, label


if __name__ == "__main__":

    root_path = "/home/tomrunia/data/VideoCountingDataset/BlenderSyntheticRandom/videos_as_dataset"
    dataset = BlenderSyntheticDataset(root_path)

    import cv2
    for i in range(10):
        frames, label = dataset.__getitem__(np.random.randint(0, len(dataset)))
        for frame in frames:
            cv2.imshow("frame", frame)
            cv2.waitKey(30)


