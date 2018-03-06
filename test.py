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


import time
from datetime import datetime
import argparse

import numpy as np
import cv2

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torchvision.utils

import torch.utils.data
import torchvision.transforms
from torch.utils.data.sampler import SubsetRandomSampler
from video_transform.spatial import *

from dataset import BlenderSyntheticDataset

from models.conv3d import Conv3D_Repetition
from dataset import init_datasets
from utils import *

################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train 3D ConvNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument('--data_path', type=str, required=True, help='Root path for dataset.')
    parser.add_argument('--checkpoint_file', type=str, required=True, help='Root path for dataset.')

    # Optimization options
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--valid_frac', type=float, default=0.1, help='Fraction of dataset to use for validation.')
    parser.add_argument('--inspect', type=bool, default=False, help='Whether to inspect classification results.')

    # Acceleration
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs. Set to 0 to perform on CPU.')
    parser.add_argument('--workers', type=int, default=8, help='Pre-fetching threads.')

    args = parser.parse_args()

    ############################################################################

    valid_transform = Compose([
        torchvision.transforms.ToPILImage(),
        ToTensor(225),
        Normalize([0, 0, 0], [1, 1, 1])
    ])

    valid_data = BlenderSyntheticDataset(args.data_path, spatial_transform=valid_transform)

    # Determine which examples to use for training and validation
    num_examples = len(valid_data)
    indices = np.arange(0, num_examples)
    split = int(np.floor(args.valid_frac * num_examples))

    # Set the indices for training and validation
    valid_idx = indices[:split]
    valid_sampler = SubsetRandomSampler(valid_idx)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=args.batch_size, sampler=valid_sampler,
        num_workers=args.workers, pin_memory=True)

    ############################################################################

    # Define the network
    net = Conv3D_Repetition(num_classes=valid_data.num_classes())

    print("Restoring weights from checkpoint: {}".format(args.checkpoint_file))
    assert os.path.exists(args.checkpoint_file), "Checkpoint does not exist."

    checkpoint = torch.load(args.checkpoint_file)
    net.load_state_dict(checkpoint['state_dict'])
    print("Succesfully restored checkpoint.")

    if args.num_gpu > 0:
        net.cuda()

    net.eval()

    # Loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    ############################################################################

    num_batches = len(valid_loader)
    batch_losses = np.zeros(num_batches)
    batch_accuracies = np.zeros(num_batches)

    for i, (inputs, labels) in enumerate(valid_loader):

        inputs = Variable(inputs)
        labels = Variable(labels)
        if args.num_gpu > 0:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Forward pass through the network
        logits = net(inputs)

        # Calculate and save metrics
        loss = criterion(logits, labels)
        acc = calculate_accuracy(logits, labels)

        batch_losses[i] = loss.data[0]
        batch_accuracies[i] = acc

        print("[{}] Performing validation {:04d}/{:04d}, "
              "Accuracy = {:.3f}, Loss = {:.3f}".format(
            datetime.now().strftime("%A %H:%M"), i, num_batches,
            acc, loss.data[0]
        ))

        ########################################################################
        # NumPy inspection of what examples are misclassified

        if args.inspect == True:

            videos      = np.squeeze(inputs.data.cpu().numpy(), axis=1)
            targets     = labels.data.cpu().numpy()
            predictions = np.argmax(logits.data.cpu().numpy(), axis=1)

            for example_idx in range(args.batch_size):
                correct = (targets[example_idx] == predictions[example_idx])
                print("Example {}. Target = {}. Prediction = {}. Correct = {}".format(
                    example_idx, targets[example_idx], predictions[example_idx], correct
                ))
                for frame_idx in range(videos.shape[1]):
                    frame_norm = np.zeros_like(videos[example_idx,frame_idx,:])
                    cv2.normalize(videos[example_idx,frame_idx,], frame_norm, 0.0, 1.0, cv2.NORM_MINMAX)
                    cv2.imshow("Frame", frame_norm)
                    key = cv2.waitKey(0)
                    if key == ord('n'):
                        break

        ########################################################################

    total_loss = float(np.mean(batch_losses))
    total_acc  = float(np.mean(batch_accuracies))

    print("VALIDATION SUMMARY ({} batches):".format(len(valid_loader)))
    print("  Loss:     {:.3f}".format(total_loss))
    print("  Accuracy: {:.3f}".format(total_acc))
    print("#"*60)



