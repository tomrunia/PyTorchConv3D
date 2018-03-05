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
# Date Created: 2018-03-02

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import shutil

################################################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, history=10):
        self.history = history
        self.values = np.zeros(self.history, dtype=np.float32)
        self.num_recorded = 0

    def push(self, value):
        assert np.isscalar(value)
        if self.num_recorded == 0:
            self.values.fill(value)
        else:
            self.values[:-1] = self.values[1:]
            self.values[-1] = value
        #print(self.values)
        self.num_recorded += 1

    def last(self):
        return self.values[-1]

    def average(self):
        return np.mean(self.values)

    def reset(self):
        self.values = np.zeros(self.history, dtype=np.float32)
        self.num_recorded = 0

################################################################################

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]
    return n_correct_elems / batch_size

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')