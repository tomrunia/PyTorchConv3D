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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Conv3D_Repetition(nn.Module):

    def __init__(self, num_classes=10, drop_rate=0.5):

        super(Conv3D_Repetition, self).__init__()

        self.num_classes = num_classes
        self.drop_rate = drop_rate

        ########################################################################

        self.relu  = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(1, 10, kernel_size=3, stride=1, bias=False)
        self.bn1   = nn.BatchNorm3d(10)

        self.conv2 = nn.Conv3d(10, 32, kernel_size=3, stride=1, bias=False)
        self.bn2   = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, bias=False)
        self.bn3   = nn.BatchNorm3d(64)

        self.fc4   = nn.Linear(62720, 256, bias=False)
        self.bn4   = nn.BatchNorm1d(256)

        self.fc5   = nn.Linear(256, 256, bias=False)
        self.bn5   = nn.BatchNorm1d(256)

        self.output = nn.Linear(256, self.num_classes, bias=False)

        ########################################################################

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = F.max_pool3d(x, kernel_size=(2,2,2), stride=(2,2,2))
        x = self.bn1(x)


        x = self.conv2(x)
        x = self.relu(x)
        x = F.max_pool3d(x, kernel_size=(2,2,2), stride=(2,2,2))
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = F.max_pool3d(x, kernel_size=(2,2,2), stride=(1,2,2))
        x = self.bn3(x)

        x = x.view(x.size(0), -1) #-1, self.num_flat_features(x))
        x = self.fc4(x)
        x = self.relu(x)
        x = self.bn4(x)

        x = self.fc5(x)
        x = self.relu(x)
        x = self.bn5(x)

        x = F.dropout(x, p=self.drop_rate, training=self.training, inplace=True)

        return self.output(x)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:] # remove batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

